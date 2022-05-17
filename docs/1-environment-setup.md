# Environment Setup

## 1. Install Miniconda

1.1. From the [official Conda documentation](https://docs.conda.io/en/latest/miniconda.html#windows-installers) download the Windows installer for **Python version 3.7**.

1.2. Once the download is complete, double-click the `.exe` file. You should see this installer window:

![01](images/01.png)

1.3. Use the default options for the installation. (The 'Destination Folder' may be left as the default recommended in your system. The screenshot below is just an example.)

![02](images/02.png)

![03](images/03.png)

![04](images/04.png)

1.4. When installation is finished, from the Start menu, open Anaconda Prompt.

1.5. To test the installation, run the following command `conda list`. You should see an output similar to the following:

![05](images/05.png)

## 2. Set up conda environment for training & inference on CPU

2.1. Extract `life3-initial-release.zip` to a folder, e.g. `C:\ai_engine`. You should see the following folder structure. (Some files seen below may not be present in the initial deployment package.)

![07](images/07.png)

2.2. Open Anaconda Prompt and change the working directory to the location where the files were extracted using the following command:
```
cd C:\ai_engine
```

2.3. Enter the following command to create a new conda environment and install the relevant dependencies required by both the training & inference modules:
```
conda env create -f life3-biotech-conda-env.yml
```

2.4. Once the environment creation is complete, run `conda env list` and verify that the environment named `life3-biotech` is listed.

2.5. Run `conda activate life3-biotech` to activate the environment. It should now appear as the activated environment as per the screenshot below. The Conda environment setup is now complete.

![08](images/08.png)

## 3. Set up files required by AI engine

3.1. If this is the first time the AI engine is being set up in the current environment, run the following command (in the same terminal window) to build the Cython files required by EfficientDet:
```
python src/life3_biotech/modeling/EfficientDet/setup.py build_ext --inplace
```
You should see an output similar to the following:

![09](images/09.png)

You may also see a `build` subfolder created in the working directory, e.g. `C:\ai_engine\build`.

3.2. In order to use the training module, data must exist in the `\data` subfolder. Copy the training data into its respective subfolder in the `\data\uploaded` subfolder. An example of the folder structure is as follows:

![10](images/10.png)

## 4. Set up Windows environment variables

4.1. Search for "environment variables" in the Windows search bar. Open "Edit the system environment variables" under Control Panel.

![11](images/11.png)

4.2. The following window will appear. Click on the "Environment Variables..." button.

![12](images/12.png)

4.3. The following pop-up dialog will appear. Click on the "New..." button below the "System variables" list.

![13](images/13.png)

4.4. The following pop-up dialog will appear. In the 'Variable name' field, enter `PYTHONPATH`. Assuming the code base has been extracted to `C:\ai_engine`, in the 'Variable value' field, enter `C:\ai_engine\src;C:\ai_engine\src\life3_biotech\modeling\EfficientDet`. Click "OK".

4.5. You should see the new variable reflected in the "System variables" list. Click "OK".

![15](images/15.png)
