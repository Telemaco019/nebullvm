from typer import Typer

app = Typer()


@app.command()
def foo():
    pass


def main():
    app()
