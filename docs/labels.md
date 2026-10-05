# Label definitions

The classifier uses three labels. The generator assigns them from these rules. This file is the definition the data and the reports follow.

## Supported

The evidence clearly covers the claim. A reader who trusts the evidence would accept the claim. Paraphrase is fine. Postgres can support a PostgreSQL claim. A longer tenure can support an at least N years claim.

Example claim: At least 5 years of PostgreSQL experience.
Example evidence: The backend engineer used Postgres for 7 years at Harbor Analytics.
Example label: Supported

## Partially Supported

The evidence covers only part of the claim, or it is related but weaker than a full match. Use this when one of two requirements is present and the other is absent. Use this when the technology matches but the years are fewer than the claim. Use this when the claim asks for production work and the evidence is a course, a class project, or a short prototype.

Example claim: At least 6 years of Kafka experience.
Example evidence: The data engineer used Kafka for 2 years at Lumen Retail.
Example label: Partially Supported

## Not Evidenced

The evidence does not support the claim. This includes hard negatives where a related technology appears. Pairs in this dataset include AWS and GCP, Docker and Kubernetes, PostgreSQL and MySQL, React and Angular, and other near misses of the same kind. It also includes evidence that names the claimed tool only to say it was rejected, and evidence about a different kind of work entirely.

Example claim: The candidate must have production experience with PostgreSQL.
Example evidence: At Copper Street Bank, the backend engineer ran MySQL in production for 5 years and owned live incidents.
Example label: Not Evidenced

A matching number of years on the wrong tool is still Not Evidenced. The years do not transfer across a near miss.
