import click

from paocli.cli import pass_environ
from paocli.utils.doctor import run_doctor


@click.command(
    name="doctor", short_help="Check native deps for running pao-tools without Docker."
)
@click.option(
    "--strict",
    is_flag=True,
    show_default=True,
    help="Exit nonzero when a required dependency is missing.",
)
@pass_environ
def cli(ctx, strict):
    """
    \b
    paocli doctor
    paocli doctor --strict
    """
    report = run_doctor()
    click.secho("paocli doctor --> os: {}".format(report["os"]), fg="cyan")
    for check in report["checks"]:
        mark = "OK  " if check["ok"] else ("MISS" if check["required"] else "opt ")
        color = "green" if check["ok"] else ("red" if check["required"] else "yellow")
        click.secho(
            "  [{}] {} -- {}".format(mark, check["name"], check["detail"]), fg=color
        )
        if not check["ok"]:
            ctx.vlog("doctor ---> {}: {}".format(check["name"], check["hint"]))
            if check["required"]:
                click.secho("         hint: {}".format(check["hint"]), fg="red")
    if report["ok_required"]:
        click.secho(
            "paocli doctor --> required deps present ({}/{}).".format(
                report["required_total"], len(report["checks"])
            ),
            fg="green",
        )
    else:
        click.secho(
            "paocli doctor --> missing required: {}".format(
                ", ".join(report["missing_required"])
            ),
            fg="red",
            err=True,
        )
        click.secho(
            "  run ./scripts/setup-native.sh, then re-run paocli doctor.", err=True
        )
        if strict:
            raise click.Abort()
