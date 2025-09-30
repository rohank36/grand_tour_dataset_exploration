import kleinkram
from pathlib import Path

def pretty_print_missions(missions):
    for m in missions:
        print(f"{m.id} | {m.name} \n")

def bytes_to_gb(bytes):
    return bytes / (1024**3)


download_folder = "tmp/grand_tour_data" 
Path(download_folder).mkdir(parents=True, exist_ok=True)

missions = kleinkram.list_missions(project_names=["GrandTourDataset"])
#pretty_print_missions(missions)
#print(len(missions))

#patterns = ["*hdr_left.bag", "*_anymal_state.bag"] # Regex patterns to match file names or do ["*"] for all files.
patterns = ["*"]

vaild_mission_names = []
for m in missions:

    filepath = Path(f"{download_folder}/{m.id}")
    filepath.mkdir(exist_ok=True)
    files = kleinkram.list_files(mission_ids=[m.id], file_names=patterns)

    with open(f"{filepath}/{m.id}.txt","w", encoding="utf-8") as fh:
        for f in files:
            fh.write(f"{f.id} | {f.name} | {f.type_} | {bytes_to_gb(f.size):.4f} GB\n")
        fh.write(f"Total Files: {len(files)}\n")

    if len(files) == 0:
        print(f"No files found for mission {m.name} matching patterns {patterns}")
        vaild_mission_names.append(m.name)
        continue
    break
    kleinkram.download(
        file_ids=[f.id for f in files],
        dest=download_folder,
        verbose=True,
        overwrite=True
    )

print(f"Failed mission names: {vaild_mission_names}")
"""
"""

