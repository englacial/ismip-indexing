from ast import Tuple
import coiled
import icechunk
import ismip6_helper
import obstore
import xarray as xr

from virtualizarr import open_virtual_dataset, VirtualiZarrDataTreeAccessor
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

import zarr

parser = HDFParser()
bucket = "gs://ismip6"
store = obstore.store.from_url(bucket, skip_signature=True)
registry = ObjectStoreRegistry({bucket: store})

@coiled.function(
    region="us-west1",
    keepalive="1 hour", 
    spot_policy="on-demand",   
)
def files_dataframe():
   return ismip6_helper.build_file_index()

files_df = files_dataframe()
files = files_df.to_dict('records')
print("Step 1/3 Complete: Built file index!")

@coiled.function(
    region="us-west1",
    keepalive="1 hour",
    spot_policy="on-demand",
)
def virtualize_file(row) -> Tuple(str, xr.Dataset):
    p = f'{row["institution"]}_{row["model_name"]}/{row["experiment"]}/{row["variable"]}' # DataTree path
    
    vds = open_virtual_dataset(
        url=row["url"],
        parser=parser,
        registry=registry,
        loadable_variables=['time', 'x', 'y'],
        decode_times=False
    )
    vds_fix_time = ismip6_helper.fix_time_encoding(vds)
    vds_fix_coords = ismip6_helper.correct_grid_coordinates(vds_fix_time, row["variable"])
    # key, dataset tuble
    return (p, vds_fix_coords)

# Map over all files in parallel
all_vdss = virtualize_file.map(files)

for vds in all_vdss:
    print(f"finished {vds[0]}")

print("Step 2/3 Complete: Done virtualizing files!")

# Now combine
@coiled.function(
    region="us-west1",
    keepalive="1 hour",
    memory="128 GB",
    cpu=16,
    spot_policy="on-demand",
)
def combine_and_write(refs_list):
    zarr.config.set({
        'async': {'concurrency': 100, 'timeout': None},
        'threading': {'max_workers': None}
    })    
    # refs list is list of tuples [(key, dataset), ...]
    # first create the dict
    datasets = {k: dataset for k, dataset in refs_list}
    # then create the data tree
    ismip6_dt = xr.DataTree.from_dict(datasets)
    # then the virtualizarr accessor
    vzdt = VirtualiZarrDataTreeAccessor(ismip6_dt)
    # then write to icechunk
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )

    # When opening the repo, use None for anonymous/public access
    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None  # None uses anonymous credentials if allowed
    })

    icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="12042025", from_env=True)
    repo = icechunk.Repository.create(icechunk_storage, config, authorize_virtual_chunk_access=credentials)
    session = repo.writable_session("main")
    vzdt.to_icechunk(session.store)
    return print(session.commit("Createed virtual store"))

print("Starting combine and write")
combine_and_write(list(all_vdss))
print("Step 3/3 Complete: Combined virtual datasets to data tree and wrote icechunk store!")
