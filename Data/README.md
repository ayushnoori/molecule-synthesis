# Data
## Download Data
### Download SyntheMol data dependencies
Follow the instructions under the **Process Data** in the `SyntheMol/README.md` file to download the basic data dependencies for the SyntheMol code.

> **Note:** Use the `Data/synthesis_data` directory in the root of the `molecule-synthesis` repository instead of the `data` directory in the `SyntheMol` directory.

### Download Blood-Brain Barrier Database (B3DB)
Download the benchmark dataset, Blood-Brain Barrier Database (B3DB) from the [GitHub repository](https://github.com/theochem/B3DB/tree/main/B3DB) and put it in the `Data/B3DB` directory. A fast way to do this is to go to [DownGit](https://minhaskamal.github.io/DownGit/#/home?url=https:%2F%2Fgithub.com%2Ftheochem%2FB3DB%2Ftree%2Fmain%2FB3DB) and click on "Download".

Then extract the contents of the database by running the following commands:
```bash
cd Data/B3DB
gunzip B3DB_classification_extended.tsv.gz
gunzip B3DB_regression_extended.tsv.gz
```

This will extract the extended version of the B3DB database, which contains additional columns with calculated chemical descriptors.