# AL-Path-Dependent

This repository contains code for the online experimentation procedure and
path-dependent acquisition functions for the paper "Active Learning Path-Dependent
Properties Using a Cloud-Based Materials Acceleration Platform".

## Installation

To use the code in this repo, install the following packages via `pip`.

```bash
python -m pip install --upgrade pip
pip install httpx==0.25.0
pip install pydantic==1.10.12
pip install 'pydantic[dotenv]==1.10.12'
pip install numpy jupyter notebook tqdm matplotlib pandas scipy scikit-learn
pip install data-request-client --upgrade --extra-index-url https://pypi.modelyst.com --upgrade --no-cache
```

## Before Running

Before running the live client for online active learning, it is a good idea to inspect
the current client queue, to ensure there are no current pending requests. This can be
done in Python, by executing the following:

```python
from data_request_client.client import DataRequestsClient

client = DataRequestsClient()

with client:
    requests = client.read_data_requests(status="completed")

print(f'There are {len(requests)} items in the requests queue:')
for tup in [(idx, r.status) for idx, r in enumerate(requests)]:
    print(f'{tup[0]}: Status - {tup[1]}')
```

This will print a list of current requests and status values. **Note:** if there are any
current pending requests in the queue, these will be queried first before actively
chosen queries.

## Example Usage

To run this code for online experimentation involving the path-dependent acquisition
functions, see the example runner scripts in the [`examples/`](examples/) directory.

For example, to run active data collection on plate 6475 (NiCo composition space, pH 1.8),
execute:

```bash
python examples/run_active_learning.py
```
