from client import CreateDataRequestModel, DataRequestsClient, UpdateDataRequestModel


# Script to test the live client.


# Create a client instance
client = DataRequestsClient()


if __name__ == "__main__":
    with client:
        # Create 4 data request
        requests = []
        for i in range(4):
            output = client.create_data_request(
                item=CreateDataRequestModel(
                    composition={"Fe": 1 - 0.1 * i, "Ni": 0.1 * i},
                    sample_label=f"sample_{i}",
                    score=i,
                )
            )
            requests.append(output)

        print(f"Data request ids: {[item.id for item in requests]}")
        request_id = requests[0].id
        # Get the data request by id
        output = client.read_data_request(output.id)
        print(f"Data request status: {output.status}")
        # Get score orders
        output = client.read_data_requests(status="pending")
        print(f"Data request scores: {[item.score for item in output]}")
        # Update the data request
        output = client.update_data_request(
            item=UpdateDataRequestModel(id=request_id, score=10)
        )

        output = client.read_data_request(output.id)
        print(f"Data request score: {output.score}")

        # Read all data requests
        output = client.read_data_requests()

        print(f"Data request count: {len(output)}")
        print(f"Data request ids: {[item.id for item in output]}")

        # get pending data requests
        output = client.read_data_requests(status="pending")
        print(f"Pending data request count: {len(output)}")

        # get order of scores
        print(f"Data request scores: {[item.score for item in output]}")

        # update request to put at front
        output = client.update_data_request(
            item=UpdateDataRequestModel(id=output[0].id, score=1000)
        )
        # get pending data requests
        output = client.read_data_requests(status="pending")
        print(f"The first data request score: {output[0].score}")

        # Acknowledge the data request
        output = client.acknowledge_data_request(request_id)
        print(f"Data request status: {output.status}")

        # get all ids
        output = client.read_data_requests(status="pending")
        print(f"Data request count: {len(output)}")

        # delete all data requests
        for item in output:
            print(f"Deleting data request {item.id}")
            output = client.delete_data_request(
                item.id,
            )

        # get all ids
        output = client.read_data_requests()
        print(f"Data request count: {len(output)}")
        print(f"Output: {output}")
