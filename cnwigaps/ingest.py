# ingest
from shutil import copytree
from pathlib import Path

prefix = "aoi_"
exclued_dirs = ["Plots", "RELABELED", "data"]
files = []

output = Path("/home/rhamilton/code/cnwigaps/test_data/aoi_data")

"""need to keep track of the aoi_name and all of the region_ids data files"""

dest = []
for x in root.iterdir():
    data = {}
    if x.name.startswith(prefix) and x.name not in exclued_dirs:
        for y in x.iterdir():
            # this gets the aoi data directory
            if y.name == "data":
                # this gets the region id data directory
                for z in y.iterdir():
                    if z.is_dir():
                        # this gets the data files
                        aoi_dest = output / Path(x.name) / Path(z.name)
                        dest.append(aoi_dest)
                        print(
                            "Copying {} to {}".format(
                                z, output / Path(x.name) / Path(z.name)
                            )
                        )
                        copytree(
                            z, output / Path(x.name) / Path(z.name), dirs_exist_ok=True
                        )
