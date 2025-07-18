import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from activelearning.acquisition import get_sample_id_comp, Acquisition


def get_valid_sic_dict(plate_id_, data_file="data/NiCoSb_active_learning_samples_241104.json"):

    sic_dict, valid_sic_dict = get_sample_id_comp(plate_id_, data_file)
    print(f"Length of valid_sic_dict = {len(valid_sic_dict)}")

    return valid_sic_dict

def make_and_get_acq(
    valid_sic_dict,
    plate_ph,
    previous_plate_json_list,
    previous_plate_ph_list,
    live_client=False,
    already_sampled=[]
):
    acq = Acquisition(
        valid_sic_dict,
        plate_ph,
        previous_plate_json_list=previous_plate_json_list,
        previous_plate_ph_list=previous_plate_ph_list,
        live_client=live_client,
    )

    # Remove already_sampled
    for sample_id in already_sampled:
        try:
            acq.ids_left.remove(sample_id)
        except:
            print(f"Not in sample_id list: {sample_id}")

    print(f"Length of acq.ids_left = {len(acq.ids_left)}\n")

    return acq


# Specify configuration for next experiment
plate_id = 6475
plate_ph = 1.8
valid_sic_dict = get_valid_sic_dict(6475, data_file="data/NiCoSb_active_learning_samples.json")

# Specify prior plate dataset
previous_plate_json_list = [
    "data/data_20240819155113.json",
]
previous_plate_ph_list = [1]

# Instantiate Acquisition object
acq = make_and_get_acq(
    valid_sic_dict, plate_ph, previous_plate_json_list, previous_plate_ph_list
)

# Run active learning
acq.run_loop(n_iter=len(acq.ids_left))
print("Experiment script complete.")
