# Error analysis

Run name: full
Test rows: 180
Missed rows: 22

This file quotes test rows the saved adapter got wrong. The quotes are copied from the test file.

Misses by true label and predicted label:
Supported predicted as Partially Supported: 5
Supported predicted as Not Evidenced: 10
Not Evidenced predicted as Supported: 3
Not Evidenced predicted as Partially Supported: 4

## Miss 1

True label: Not Evidenced
Predicted label: Partially Supported
Predicted probability: 0.4627
Claim: At least 4 years of PostgreSQL experience.
Evidence: Resume line from Aster Payroll: 4 years with MySQL as the data engineer.

The evidence does not support the claim, but the model saw a weak relation and called it partial.

## Miss 2

True label: Not Evidenced
Predicted label: Supported
Predicted probability: 0.5233
Claim: Required tenure: 6 years using Docker.
Evidence: K8s was the daily tool for this infrastructure engineer for 6 years at Oakline Energy.

The evidence does not support the claim, but the model called it full support. A related tool or a shared phrase can pull the score that way.

## Miss 3

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.5477
Claim: Required: operate Jenkins under live traffic.
Evidence: The production stack at Brightline Health centered on Jenkins. The analytics engineer owned it for 5 years.

The evidence covers the claim, but the model found no support.

## Miss 4

True label: Supported
Predicted label: Partially Supported
Predicted probability: 0.5566
Claim: Required tenure: 5 years using DynamoDB.
Evidence: At Fieldnote Media, the data engineer spent 6 years working with Dynamo.

The evidence covers the claim, but the model called it only partial.

## Miss 5

True label: Not Evidenced
Predicted label: Partially Supported
Predicted probability: 0.8838
Claim: This role needs RabbitMQ together with writing design documents for the team.
Evidence: The platform engineer did not adopt RabbitMQ. Work at Fieldnote Media stayed on Kafka.

The evidence does not support the claim, but the model saw a weak relation and called it partial.

## Miss 6

True label: Not Evidenced
Predicted label: Supported
Predicted probability: 0.4902
Claim: At least 8 years of Apache experience.
Evidence: At Harbor Analytics, the analytics engineer spent 8 years working with Nginx.

The evidence does not support the claim, but the model called it full support. A related tool or a shared phrase can pull the score that way.

## Miss 7

True label: Supported
Predicted label: Not Evidenced
Predicted probability: 0.5002
Claim: Required: operate Elasticsearch under live traffic.
Evidence: At Harbor Analytics, the platform engineer ran Elastic Search in production for 2 years and owned live incidents.

The evidence covers the claim, but the model found no support.

## Miss 8

True label: Supported
Predicted label: Partially Supported
Predicted probability: 0.6183
Claim: At least 5 years of Polars experience.
Evidence: The backend engineer used Polars for 6 years at Lumen Retail.

The evidence covers the claim, but the model called it only partial.

## Miss 9

True label: Not Evidenced
Predicted label: Partially Supported
Predicted probability: 0.8471
Claim: This role needs FastAPI together with writing design documents for the team.
Evidence: FastAPI was never used in production at Summit Transit. The live stack was Spring Boot.

The evidence does not support the claim, but the model saw a weak relation and called it partial.

## How to read the misses

Supported versus Partially Supported is the finer cut: the tool may match while the years, the second requirement, or the setting does not. Not Evidenced includes near misses such as PostgreSQL versus MySQL, plus rows where the claimed tool is named and then rejected.
