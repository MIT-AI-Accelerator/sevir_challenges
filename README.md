# sevir_challenges
A collection of tasks and baseline models for the SEVIR weather dataset


## Obtaining SEVIR data

To obtian the dataset used in each of the challenges, run the following command

To download, install AWS CLI, and download all of SEVIR (~1TB) to your current directory run
```
aws s3 sync --no-sign-request s3://sevir .
```
Each of the benchmarks in this repo use a subset of the full SEVIR dataset.












