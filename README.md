# sevir_challenges
A collection of challenges and baseline models for the SEVIR weather dataset.

## Obtaining SEVIR data

The challenges in this repo are based on the [SEVIR weather dataset](https://proceedings.neurips.cc//paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html).  This dataset is made of up sequences of weather data imagery sampled and aligned from weather radar and satellite.   It was constucted as a benchmark dataset to suuport algorithm development in meterology. For more information on this dataset see [the SEVIR tutorial.](https://nbviewer.jupyter.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb)

SEVIR is currently available for download from the [AWS Open Data registry](https://registry.opendata.aws/sevir/).  In total, the dataset is approximately 1TB in size, however smaller samples of the full dataset are provided for selected challenges (see `s3://sevir/data/processed/`).

To train models using the full dataset, you can download SEVIR using one of the following methods.  

### Using AWS CLI

If you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html), you can download SEVIR using the 

```
aws s3 sync --no-sign-request s3://sevir .
```

To download only a specific modalitiy, e.g. `vil`, you can instead run

```
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv CATALOG.csv
aws s3 sync --no-sign-request s3://sevir/data/vil .
```

### Using `boto3` moduels

Using the python `boto3` modules (`conda install boto3`) you can obtain SEVIR data by first connecting to the S3 bucket:

```python
import boto3
from botocore.handlers import disable_signing
resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir')
```

Get a list of files using

```
objs=bucket.objects.filter(Prefix='')
print([o.key for o in objs])
```

And download files of interest, e.g.

```pthon
bucket.download_file('CATALOG.csv','/home/data/SEVIR/CATALOG.csv')
bucket.download_file('data/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0701_1231.h5','/home/data/SEVIR/data/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0701_1231.h5')
#... etc
```











