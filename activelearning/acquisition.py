import copy
import time
import json
import datetime
import ipdb

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm


def get_sample_id_comp(plate_id, data_path):
    df = pd.read_json(data_path)

    sample_id_comp = {}
    valid_sample_id_comp = {}
    for index, row in df.iterrows():
        if row.plate_id == plate_id:
            composition = {
                "Co": row.Co_fraction, "Ni": row.Ni_fraction, "Sb": row.Sb_fraction
            }
            sample_id_comp[row.sample_no] = composition 
            if row.thickness_within_2sig == True:
                valid_sample_id_comp[row.sample_no] = composition

    return sample_id_comp, valid_sample_id_comp


class Acquisition:

    def __init__(
        self,
        sic_dict,
        plate_ph,
        save_path=None,
        previous_plate_json_list=None,
        previous_plate_ph_list=None,
        live_client=False,
        queried_data_path=None,
        normalize_id = 2,
        filter_id = 2,
        max_size = 5000,
        max_acq = 500,
    ):
        self.sic_dict = sic_dict
        self.ids_left = copy.deepcopy(list(sic_dict.keys()))
        self.plate_ph = plate_ph
        self.inputs = []
        self.outputs = []
        self.live_client = live_client
        self.max_size = max_size
        self.max_acq = max_acq
        self.normalize_id = normalize_id
        self.filter_id = filter_id

        if self.live_client:
            from data_request_client.client import CreateDataRequestModel, DataRequestsClient
            self.client = DataRequestsClient()
        else:
            self.client = None

        if not save_path:
            save_path = f"data/data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        self.save_path = save_path

        if not previous_plate_json_list:
            previous_plate_json_list = None
        self.previous_plate_json_list = previous_plate_json_list

        if not previous_plate_ph_list:
            previous_plate_ph_list = None
        self.previous_plate_ph_list = previous_plate_ph_list

        self.data_current = None
        self.set_data_previous()

        if not queried_data_path:
            queried_data_path =  None
        else:
            #TODO for automatic resuming jobs, can populate self.data_current using
            # self.queried_data_path above. Also update self.ids_left to remove
            # sampled_ids of queried points
            pass
        self.queried_data_path = queried_data_path

        self.label_mean = None
        self.minmax_list = None
        self.set_ts()

    def run_loop(self, n_iter, uuid_str=None):
        if uuid_str:
            self.wait_for_output(uuid_str)

        # Loops through take_step n_iter times
        for step_idx in range(n_iter):
            self.take_step()

        # Save final data
        self.save_data()

    def take_step(self):
        self.set_data()
        query_dict = self.optimize_acquisition()
        uuid_str = self.query_function(query_dict)
        self.wait_for_output(uuid_str)
        self.save_data()

    def optimize_acquisition(self):
        print("Computing and optimizing acquisition function")
        self.fit_gp_on_data()
        sample_id, parameters = self.optimize_acquisition_over_comps()
        query_dict = {
            "sample_id": sample_id,
            "comp": self.sic_dict[sample_id],
            "parameters": parameters,
        }
        return query_dict

    def optimize_acquisition_over_comps(self):
        candidate_id_list = self.get_candidate_id_list()
        opt_input_list = []
        acq_val_list = []
        for sample_id in candidate_id_list:
            opt_input, acq_val = self.optimize_acquisition_for_one_comp(sample_id)
            opt_input_list.append(opt_input)
            acq_val_list.append(acq_val)

        opt_idx = np.argmax(acq_val_list)
        opt_input = opt_input_list[opt_idx]

        next_sample_id = opt_input["sample_id"]
        next_parameters = opt_input["parameters"]
        self.ids_left.remove(next_sample_id)
        return next_sample_id, next_parameters

    def set_data(self):
        data_current = self.data_current
        data_previous = self.data_previous
        combined_data = None

        if data_previous is not None and data_current is not None:
            combined_data = np.concatenate((data_previous, data_current))
        elif data_previous is None and data_current is not None:
            combined_data = data_current
        elif data_previous is not None and data_current is None:
            combined_data = data_previous

        if combined_data is not None:
            combined_data = self.process_data(
                combined_data, self.normalize_id, self.filter_id
            )

        if combined_data.shape[0] > self.max_size:
            keep_indices = np.sort(
                np.random.choice(
                    combined_data.shape[0], size=self.max_size, replace=False
                )
            )
            combined_data = combined_data[keep_indices]

        self.data = combined_data

    def get_data_previous_sets(self):
        if self.previous_plate_json_list is None:
            return None

        prev_data_list = []
        zip_json_ph = zip(self.previous_plate_json_list, self.previous_plate_ph_list)
        for idx, (json_path, ph_val) in enumerate(zip_json_ph):
            print(f"*** idx = {idx}, json_path = {json_path}, ph_val = {ph_val}")
            data = self.get_data_matrix(json_path, ph_val)
            print(f"data.shape = {data.shape}")
            prev_data_list.append(data)

        prev_data_concat = np.concatenate(prev_data_list)

        return prev_data_concat

    def set_data_previous(self):
        print("\nSetting previous data.")
        data_previous = self.get_data_previous_sets()
        self.data_previous = data_previous
        print("Finished setting previous data.\n")

    def query_function(self, query_dict):
        sample_id = query_dict["sample_id"]
        comp = query_dict["comp"]
        parameters = query_dict["parameters"]

        if self.live_client:
            with self.client:
                request = self.client.create_data_request(
                    item=CreateDataRequestModel(
                        composition={"Ni": comp["Ni"], "Co": comp["Co"], "Sb": comp["Sb"]},
                        sample_label=f"sample_{sample_id}",
                        score=0.5,
                        analysis=None,
                        parameters=parameters,
                    )
                )

            uuid_str = str(request.id)
        else:
            uuid_str = str(0)

        query_dict["uuid"] = uuid_str
        self.inputs.append(query_dict)
        print(f"Made query: sample_id={sample_id}, composition={comp}, parameters={parameters}")
        return uuid_str

    def wait_for_output(self, uuid_str):
        status = "pending"
        print(f"Status = {status}. Waiting for output.")
        while status != "completed":
            if self.live_client:
                with self.client:
                    output = self.client.read_data_request(uuid_str)
                status = str(output.status)
            else:
                status = "pending"

            time.sleep(10)
            print(f"Status = {status}. Waiting for output.")


        # Download output_data into analysis_dict_list
        if self.live_client:
            analysis_dict_list = []
            with self.client:
                for v in output.analysis:
                    di = {
                        "analysis_uuid": v["analysis_uuid"],
                        "z": v["process_params"]["CA_potential_vsRHE"],
                    }
                    array_data = self.client.download_output(
                        data_request_id=output.id,
                        analysis_uuid=di["analysis_uuid"],
                        output_name="array",
                    )
                    di["array_data"] = array_data
                    analysis_dict_list.append(di)

            output_dict = {
                'label': output.sample_label,
                'comp': output.composition,
                'analysis': analysis_dict_list,
            }
        else:
            output_dict = {
                'label': None,
                'comp': None,
                'analysis': None,
            }

        self.outputs.append(output_dict)
        len_in, len_out = len(self.inputs), len(self.outputs)
        print(f"Received output. Now: len(inputs) = {len_in}, len(outputs)={len_out}")

        data = []
        for out in self.outputs:
            data_sub = self.extract_data_arr_from_output(out, self.plate_ph)
            data += data_sub

        data_arr = np.array(data)
        self.data_current = data_arr

    def get_mean_ratio_term(self, output_dict, zidx):
        arr_insitu = np.array(
            output_dict["analysis"][zidx]["array_data"]["smth_insitu"]
        )
        arr_baseline = np.array(
            output_dict["analysis"][zidx]["array_data"]["smth_baseline"]
        )
        om_arr_insitu = 1 - arr_insitu
        om_arr_baseline = 1 - arr_baseline

        old_ratio_term = om_arr_insitu / om_arr_baseline
        ratio_term = arr_insitu / arr_baseline

        mean_ratio_term = np.mean(ratio_term)

        return mean_ratio_term

    def save_data(self):
        data = {"inputs": self.inputs, "outputs": self.outputs}
        if self.live_client:
            with open(self.save_path, 'w') as f:
                json.dump(data, f)
            print(f"Saved: {self.save_path}")
        else:
            print(f"did not save: {self.save_path}")

    def fit_gp_on_data(self):
        X = self.data[:, :5]
        y = self.data[:, 5]

        gp = self.get_fit_gp(X, y)
        self.gp = gp

    def get_fit_gp(
            self,
            X_tr,
            y_tr,
            transform=False,
            normalize_y=False,
            use_constant_kernel=False,
            constant_mean=None,
            constant_bounds=None,
        ):

        if use_constant_kernel:
            kernel = ConstantKernel(constant_mean, constant_bounds) * RBF(0.5, "fixed")
        else:
            kernel = RBF(0.5, "fixed")

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=normalize_y,
            alpha=1e-2,
        )

        if transform:
            gp.fit(X_tr, y_tr) #TODO add squashing version of GP normalization
        else:
            gp.fit(X_tr, y_tr)

        return gp

    def get_ieig_down(self, v_idx, pi_list, std_list):
        ieig_list = []
        ipi_list = [(1-pi) for pi in pi_list]
        eig_arr = np.log(np.array(std_list) + 10)
        for idx in list(range(v_idx + 1)):
            eig = eig_arr[idx]
            if idx > 0:
                ieig = ipi_list[idx] * (ieig_list[idx-1] + eig)
            else:
                ieig = ipi_list[idx] * eig
            ieig_list.append(ieig)
        ieig = ieig_list[-1].item()
        return ieig

    def get_ieig_up(self, v_idx, pi_list, std_list):
        ieig_list = []
        ipi_list = [(1-pi) for pi in pi_list]
        eig_arr = np.log(np.array(std_list) + 10)
        for idx in list(range(10, v_idx-1, -1)):
            eig = eig_arr[idx]
            if idx < 10:
                ieig = ipi_list[idx] * (ieig_list[-1] + eig)
            else:
                ieig = ipi_list[idx] * eig
            ieig_list.append(ieig)
        ieig = ieig_list[-1].item()
        return ieig

    def get_pi_like(self, mean, std):
        pi_upper = norm.sf(1.15, loc=mean, scale=std)
        pi_lower = norm.cdf(0.85, loc=mean, scale=std)
        pi_like = pi_upper + pi_lower
        return pi_like

    def optimize_acquisition_for_one_comp(self, sample_id):
        pred_list, std_list = self.get_pred_and_std_for_comp(sample_id)
        pi_list = self.get_pi_list_for_comp(sample_id, pred_list, std_list)
        ieig_down_list = [self.get_ieig_down(vidx, pi_list, std_list) for vidx in range(11)]
        idx_opt_ieig_down = np.argmax(ieig_down_list)
        ieig_up_list = [self.get_ieig_up(vidx, pi_list, std_list) for vidx in range(11)]
        idx_opt_ieig_up = np.argmax(ieig_up_list)
        z_list = np.around(np.linspace(0.0, 2.0, 11), decimals=1)
        if ieig_down_list[idx_opt_ieig_down] > ieig_up_list[idx_opt_ieig_up]:
            opt_direction = "down"
            opt_z = z_list[idx_opt_ieig_down]
            opt_acq_val = ieig_down_list[idx_opt_ieig_down]
        else:
            opt_direction = "up"
            opt_z = z_list[idx_opt_ieig_up]
            opt_acq_val = ieig_up_list[idx_opt_ieig_up]

        opt_input = {
            "sample_id": sample_id,
            "parameters": {
                "z_start": opt_z,
                "z_direction": opt_direction,
            },
        }
        return opt_input, opt_acq_val

    def get_pred_and_std_for_comp(self, sample_id):
        comp = list(self.sic_dict[sample_id].values())
        feat = np.array(comp + [0.0, self.plate_ph]).reshape(1, -1)
        z_list = np.around(np.linspace(0.0, 2.0, 11), decimals=1)
        pred_list = []
        std_list = []
        for idx in range(11):
            feat[0, 3] = z_list[idx]
            normalized_feat = self.normalize_features(feat, copy_array=True)
            y_pred, sigma = self.gp.predict(normalized_feat, return_std=True)
            y_pred = y_pred[0]
            sigma = sigma[0]
            y_ub = y_pred + sigma
            y_lb = y_pred - sigma

            pred_list.append(y_pred)
            std_list.append(sigma)

        return pred_list, std_list

    def get_pi_list_for_comp(self, sample_id, pred_list, std_list):
        pi_list = []
        label_mean = self.label_mean
        for mean, std in zip(pred_list, std_list):
            pi_like = self.get_pi_like(mean + label_mean, std)
            pi_list.append(pi_like)

        pi_arr = np.array(pi_list).reshape(-1)
        ts = np.exp(self.ts_w * np.log(self.ts) + (1 - self.ts_w) * np.log(pi_arr))
        return ts

    def get_data_matrix(self, json_file_path, ph):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        outputs = data["outputs"]
        outputs = [out for out in outputs if type(out)==dict]

        print(f"Loaded json file: {json_file_path}")
        print(f"Number outputs: {len(outputs)}")

        data = []

        for output in outputs:

            data_sub = self.extract_data_arr_from_output(output, ph)
            data += data_sub

        data = np.array(data)
        return data

    def extract_data_arr_from_output(self, output, ph):
        data = []

        for analysis in output["analysis"]:

            stoich = [float(i) for i in output["comp"].values()]
            z = float(analysis["z"])

            arr_insitu = np.array(analysis["array_data"]["smth_insitu"])
            arr_baseline = np.array(analysis["array_data"]["smth_baseline"])
            mean_ratio_term = float(np.mean(arr_insitu / arr_baseline))

            data.append(stoich + [z] + [ph] + [mean_ratio_term])

        return data

    def process_data(self, arr, normalize_id=1, filter_id=1):
        arr_orig = copy.deepcopy(arr)

        if filter_id == 1:
            arr = arr
        elif filter_id == 2:
            arr = self.label_row_stabilities(arr)
            arr = self.filter_rows_by_label(arr)
        else:
            pass

        if normalize_id == 1:
            arr = arr
        elif normalize_id == 2:
            self.set_minmax_list_and_label_mean(arr)
            arr = self.normalize_data(arr)
        elif normalize_id == 3:
            arr = self.normalize_data_ph(arr)
        else:
            pass

        return arr

    def normalize_data_ph(self, arr):
        normalized_data = arr.copy()

        col = 4
        min_val = self.minmax_list[col][0]
        max_val = self.minmax_list[col][1]
        if max_val > min_val:
            normalized_data[:, col] = (arr[:, col] - min_val) / (max_val - min_val)
        else:
            normalized_data[:, col] = 0.0

        return normalized_data

    def normalize_data(self, arr, copy_array=True):
        if copy_array:
            normalized_data = arr.copy()
        else:
            normalized_data = arr

        normalized_data = self.normalize_features(normalized_data)
        normalized_data[:, 5] = normalized_data[:, 5] - self.label_mean

        return normalized_data

    def normalize_features(self, arr, copy_array=True):
        if copy_array:
            normalized_feat = arr.copy()
        else:
            normalized_feat = arr

        for col in range(5):
            min_val = self.minmax_list[col][0]
            max_val = self.minmax_list[col][1]
            if max_val > min_val:
                normalized_feat[:, col] = (arr[:, col] - min_val) / (max_val - min_val)
            else:
                normalized_feat[:, col] = 0.0

        return normalized_feat

    def set_minmax_list_and_label_mean(self, arr):

        minmax_list = []
        for col in range(5):
            min_val = arr[:, col].min()
            max_val = arr[:, col].max()
            minmax_list.append((min_val, max_val))

        label_mean = arr[:, 5].mean()

        self.minmax_list = minmax_list
        self.label_mean = label_mean

    def set_ts(self):
        pi_prior_params = self.get_pi_prior_params()
        beta = np.array([np.random.beta(a, b) for a, b in pi_prior_params])
        self.ts = beta

    def get_candidate_id_list(self):
        ids_left = copy.deepcopy(self.ids_left)

        if self.max_acq > len(ids_left):
            sub_size = len(ids_left)
        else:
            sub_size = self.max_acq

        ids_left_sub = np.random.choice(ids_left, size=sub_size, replace=False)
        return ids_left_sub

    def get_pi_prior_params(self):
        alpha_params = [13.5, 10.5, 7.5, 5.25, 3.75, 0.75, 3.75, 5.25, 7.5, 10.5, 13.5]
        beta_params  = [1.5, 4.5, 7.5, 9.75, 11.25, 14.25, 11.25, 9.75, 7.5, 4.5, 1.5]
        self.ts_w = np.random.uniform(1 - 2e-3, 1 - 1e-6)
        pi_prior_params = list(zip(alpha_params, beta_params))
        return pi_prior_params

    def reverse_label_normalization(self, normalized_list, label_mean):
        normalized_values = np.asarray(normalized_list)
        unnormalized_values = normalized_values + label_mean

        return unnormalized_values

    def label_row_stabilities(self, arr):
        n = arr.shape[0]
        result = np.zeros((n, 7))
        result[:, :6] = arr

        current_comp = None
        instable_flag = False

        for i in range(n):
            row = arr[i]
            comp = tuple(row[:3])
            value = row[-1]

            if comp != current_comp:
                current_comp = comp
                instable_flag = False

            if not instable_flag:
                if 0.85 <= value <= 1.15:
                    result[i, -1] = 0
                else:
                    result[i, -1] = 1
                    instable_flag = True
            else:
                result[i, -1] = 2

        return result

    def filter_rows_by_label(self, arr):
        mask = (arr[:, -1] == 0.0) | (arr[:, -1] == 1.0)
        return arr[mask, :-1]
