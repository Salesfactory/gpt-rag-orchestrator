import requests

payload = {
    "equity_id": "LOW",
    "filing_type": "8-K",
    "after_date": '2024-12-20',
}

response = requests.post(
    "https://webgpt0-vm2b2htvuuclm.azurewebsites.net/api/SECEdgar/financialdocuments/process-and-summarize",
    headers={
        "Content-Type": "application/json",
        # TODO: add authorization header
    },
    json=payload
)

response_json = response.json()

print(response_json['code'])