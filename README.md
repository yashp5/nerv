# Nerv Energy

Energy-efficient, demand response AI models for EV charging to help enterprises in decarbonization.

## MVP

1. Built a model that generates an energy profile for efficiently distributing energy over a 24 hr period
2. Model has only one parameter: Day Ahead Hourly Energy Rates
3. Outputs for training are generated using a simple energy distribution script based on hourly rates
4. Using NYISO New York dataset

### API Curl Commands

**Get Token API**

```bash
curl -X 'POST' \
  'https://app.nerv.energy/token' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'grant_type=&username=testuser&password=testpassword&scope=&client_id=&client_secret='

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6NjE3Mjg0NjkxOTl9.nAzCgvF7U4RjS9b9aKio0TcQjjQOWD4LI8ffh0hTDsY",
  "token_type": "bearer"
}
```

**Get Todays Hourly Rates Data from NYISO**

```bash
curl -X 'GET' \
  'https://app.nerv.energy/fetch_pricing_data/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6NjE3Mjg0NjkyNDN9.wD41K1L_4DOvkf7EHXycBCU9F7ThzahXXp3af-4rOmY'

Response:
{
  "hourly_rates": [
    30.03,
    29.29,
    27.71,
    26.63,
    27.39,
    30.950000000000003,
    40.1,
    54.91,
    46.660000000000004,
    36.28,
    32.55,
    31.089999999999996,
    31.19,
    32.55,
    33.53,
    35.93,
    42.94,
    52.04,
    59.330000000000005,
    57.940000000000005,
    47.650000000000006,
    39.95,
    35.120000000000005,
    30.02
  ]
}
```

**Generate Energy Profile**

1. Hourly Rates
2. Total Energy (kWh): Energy to be distributed over 24 hrs
3. Curtailment Limit (kW): Maximum power level

```bash
curl -X 'POST' \
  'https://app.nerv.energy/predict/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6NjE3Mjg0NjkyNDN9.wD41K1L_4DOvkf7EHXycBCU9F7ThzahXXp3af-4rOmY' \
  -H 'Content-Type: application/json' \
  -d '{
  "hourly_rates": [
    30.03,
    29.29,
    27.71,
    26.63,
    27.39,
    30.950000000000003,
    40.1,
    54.91,
    46.660000000000004,
    36.28,
    32.55,
    31.089999999999996,
    31.19,
    32.55,
    33.53,
    35.93,
    42.94,
    52.04,
    59.330000000000005,
    57.940000000000005,
    47.650000000000006,
    39.95,
    35.120000000000005,
    30.02
  ],
  "total_energy": 50,
  "curtailment_limit": 3
}'

Response:
{
  "generated_profile": [
    2.2075517177581787,
    2.67110538482666,
    2.6209259033203125,
    2.628175973892212,
    2.797395706176758,
    1.8216593265533447,
    2.020437002182007,
    1.9192907810211182,
    1.734178066253662,
    2.0147862434387207,
    1.8553223609924316,
    2.062791585922241,
    1.7181369066238403,
    2.3976333141326904,
    2.41877818107605,
    2.543221950531006,
    2.262800693511963,
    1.6644532680511475,
    2.063533067703247,
    1.7187602519989014,
    1.5921164751052856,
    1.5066879987716675,
    1.546375036239624,
    2.213881254196167
  ]
}
```

### Architecture

![Architecture](./architecture.png)
