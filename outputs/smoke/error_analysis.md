# Error analysis

Run name: smoke
Test rows: 180
Missed rows: 120

This file quotes test rows the smoke run got wrong. The smoke run uses 32 training rows and 1 epoch. Read the misses as pipeline output, not as a study of a converged model.

Every test row was predicted as Not Evidenced. The probabilities sit near one third, so the head has not separated the classes. The quotes below are misses from that collapse.

Misses by true label and predicted label:
Supported predicted as Not Evidenced: 60
Partially Supported predicted as Not Evidenced: 60

## Miss 1

True label: Partially Supported
Predicted label: Not Evidenced
Predicted probability: 0.3915
Claim: The candidate needs Vue experience and mentoring junior engineers.
Evidence: For 7 years, the infrastructure engineer at Redwood Logistics kept Vue running for real users.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 2

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.3922
Claim: This role needs Elasticsearch together with writing design documents for the team.
Evidence: Elastic Search was the core tool at Redwood Logistics. The platform engineer wrote the design documents the team used.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 3

True label: Partially Supported
Predicted label: Not Evidenced
Predicted probability: 0.3914
Claim: This role needs Vue together with writing design documents for the team.
Evidence: Daily work for the site reliability engineer at Marble Education was Vue, including live incidents over 7 years.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 4

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.3908
Claim: Required tenure: 5 years using DynamoDB.
Evidence: At Fieldnote Media, the data engineer spent 6 years working with Dynamo.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 5

True label: Partially Supported
Predicted label: Not Evidenced
Predicted probability: 0.3926
Claim: This role needs Celery together with writing design documents for the team.
Evidence: The production stack at Summit Transit centered on Celery. The platform engineer owned it for 2 years.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 6

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.3897
Claim: At least 7 years of Angular experience.
Evidence: The frontend engineer used Angular for 7 years at Harbor Analytics.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 7

True label: Partially Supported
Predicted label: Not Evidenced
Predicted probability: 0.3885
Claim: The candidate must have production experience with Ansible.
Evidence: Used Ansible in a university class project with sample data.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 8

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.3908
Claim: The candidate must have production experience with Cloud Functions.
Evidence: For 6 years, the data engineer at Brightline Health kept Cloud Functions running for real users.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## Miss 9

True label: Partially Supported
Predicted label: Not Evidenced
Predicted probability: 0.3909
Claim: At least 8 years of CloudFormation experience.
Evidence: CloudFormation was the daily tool for this frontend engineer for 6 years at Lumen Retail.

This row is one miss from a model that predicted Not Evidenced for every test row. The probability is only a little above one third, which fits a head that has not learned the labels.

## How to read the misses

Supported versus Partially Supported is the finer cut: the tool may match while the years, the second requirement, or the setting does not. Not Evidenced includes near misses such as PostgreSQL versus MySQL, plus rows where the claimed tool is named and then rejected.
