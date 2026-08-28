"""The container definitions, checked against the code they run.

None of this builds an image — that needs a Docker daemon, and a test suite
that only passes on machines with one is a test suite people stop running.

What it does check is the thing that actually breaks: drift. A Dockerfile and a
compose file are the only places in a project where commands are written as
strings that nothing verifies. Rename a CLI flag and every one of them silently
becomes wrong, and you find out when a container exits 2 at three in the
morning rather than when you make the change.

So every command in docker-compose.yml is parsed here by the real argparse
parser, and the paths are checked to match the volumes they are supposed to
land in.
"""

from __future__ import annotations

import os
import shlex

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_PATH = os.path.join(ROOT, "docker-compose.yml")
DOCKERFILE_PATH = os.path.join(ROOT, "Dockerfile")

DB_IN_CONTAINER = "/data/gridcast.db"
MODELS_IN_CONTAINER = "/models"


@pytest.fixture(scope="module")
def compose():
    with open(COMPOSE_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def dockerfile():
    with open(DOCKERFILE_PATH, encoding="utf-8") as handle:
        return handle.read()


def _parse(argv):
    """Run a container command through the real CLI parser.

    argparse calls sys.exit on a bad argument, which is what makes this a
    useful test rather than a decorative one.
    """
    from gridcast import cli

    # main() builds the parser and then runs the command. Rebuilding the parser
    # here would test a copy of it rather than the thing itself, so every
    # command function is swapped for one that records its arguments and
    # returns — the parse is real, the side effects are not.
    captured = {}

    def capture(args):
        captured["args"] = args
        return 0

    original = {name: getattr(cli, name)
                for name in dir(cli) if name.startswith("cmd_")}
    for name in original:
        setattr(cli, name, capture)
    try:
        assert cli.main(list(argv)) == 0
    finally:
        for name, func in original.items():
            setattr(cli, name, func)

    return captured["args"]


# ------------------------------------------------------------------ compose
def test_every_compose_command_is_a_real_cli_command(compose):
    for name, service in compose["services"].items():
        command = service.get("command")
        if command is None:
            continue
        argv = command if isinstance(command, list) else shlex.split(command)
        args = _parse(argv)
        assert args.command, f"service {name!r} parsed to no subcommand"


def test_services_write_to_the_volumes_they_mount(compose):
    """A command pointed at a path with no volume behind it loses its work.

    This is the failure that looks like success: the container runs, reports
    rows written, exits 0, and the database it wrote disappears with it.
    """
    for name, service in compose["services"].items():
        command = service.get("command")
        if command is None:
            continue
        mounted = {v.split(":")[1] for v in service.get("volumes", [])}
        argv = command if isinstance(command, list) else shlex.split(command)

        if "--db" in argv:
            assert "/data" in mounted, f"{name} writes a db with no /data volume"
        if "--model-dir" in argv:
            assert "/models" in mounted, f"{name} saves models with no /models volume"


def test_api_is_the_only_service_started_by_default(compose):
    """`docker compose up` must not start a fetch against EirGrid.

    Their open data licence asks people not to hammer the service. A compose
    file where `up` triggers ingest turns every restart into another request
    burst, which is exactly the habit not to form.
    """
    default = [n for n, s in compose["services"].items() if not s.get("profiles")]
    assert default == ["api"]


def test_api_port_is_bound_to_localhost(compose):
    """"8000:8000" publishes past the host firewall — Docker writes its own
    iptables rules. Nothing here needs to be reachable from the internet."""
    for published in compose["services"]["api"]["ports"]:
        assert str(published).startswith("127.0.0.1:")


def test_one_shot_services_share_the_api_image(compose):
    images = {s["image"] for s in compose["services"].values()}
    assert len(images) == 1, "a second image is a second thing to drift"


# --------------------------------------------------------------- dockerfile
def test_container_binds_all_interfaces(dockerfile):
    """127.0.0.1 inside a container binds the container's own loopback.

    The published port then connects to nothing, and the service looks broken
    while running perfectly. This is the single most common way a working
    application fails to work in Docker.
    """
    assert "--host" in dockerfile and "0.0.0.0" in dockerfile


def test_container_does_not_run_as_root(dockerfile):
    assert "USER gridcast" in dockerfile
    user_line = dockerfile.index("USER gridcast")
    assert dockerfile.index("COPY --from=builder") < user_line, (
        "copy before dropping privileges, or the copy fails"
    )


def test_image_installs_serve_and_not_analysis(dockerfile):
    """The service answers requests; it never draws a chart.

    matplotlib plus its font and image libraries is a large thing to ship into
    an image with no use for it.
    """
    import tomllib

    assert '".[serve]"' in dockerfile
    assert ".[analysis]" not in dockerfile

    # Checked against the extra rather than against the Dockerfile text, which
    # mentions matplotlib only to explain why it is absent.
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        config = tomllib.load(handle)
    serve = " ".join(config["project"]["optional-dependencies"]["serve"])
    assert "matplotlib" not in serve


def test_serve_extra_covers_what_the_api_imports(dockerfile):
    """Whatever the image installs has to include the modelling stack.

    The API loads a joblib model and builds pandas features on every request.
    An image with fastapi but no scikit-learn starts cleanly and then fails on
    the first forecast, which is the worst possible time to find out.
    """
    import tomllib

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        config = tomllib.load(handle)

    serve = " ".join(config["project"]["optional-dependencies"]["serve"])
    for package in ("pandas", "numpy", "scikit-learn", "joblib", "fastapi", "uvicorn"):
        assert package in serve, f"{package} missing from the serve extra"


def test_joblib_is_declared_rather_than_inherited():
    """model.py imports joblib by name.

    It arrives as a scikit-learn dependency today. Relying on somebody else's
    dependency tree to supply a module you import directly breaks quietly on
    the day they drop it.
    """
    import tomllib

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        config = tomllib.load(handle)

    extras = config["project"]["optional-dependencies"]
    assert any("joblib" in dep for dep in extras["analysis"])
    assert any("joblib" in dep for dep in extras["serve"])


def test_dockerignore_excludes_the_database(dockerfile):
    """A year of readings baked into an immutable image is stale the day after."""
    with open(os.path.join(ROOT, ".dockerignore"), encoding="utf-8") as handle:
        ignored = handle.read()
    for pattern in ("data/", "models/", ".venv", "*.db"):
        assert pattern in ignored
