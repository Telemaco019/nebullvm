from rich import box
from rich import print
from rich.table import Table

from surfer.cli.commands.common import must_load_config
from surfer.computing import clusters


def list_accelerators():
    surfer_config = must_load_config()
    accelerators = clusters.get_available_accelerators(surfer_config)
    table = Table(box=box.SIMPLE, expand=False)
    table.add_column("Code", style="cyan")
    table.add_column("Description", style="cyan")
    for a in accelerators:
        table.add_row(a.value, a.display_name)
    print(table)
    if len(accelerators) == 0:
        print("No accelerators found in cluster at {}".format(
            surfer_config.cluster_file.as_posix()
        ))
