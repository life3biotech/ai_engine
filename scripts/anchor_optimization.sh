IMAGE_MIN_SIDE=512
IMAGE_MAX_SIDE=512
SCALES=5
RATIOS=5
ANNOTATION_PATH='./data/processed/annotations_train_efficientdet.csv'

# Build package
build_package() 
{
    # Download sudo dependencies
    sudo apt update
    sudo apt-get install gcc

    # Set up conda environment
    eval "$(conda shell.bash hook)"
    conda create -y -n temp python=3.6.8
    conda activate temp
    cd anchor_optimization

    # Download pip dependencies and build cython codes
    pip install .
    python setup.py build_ext --inplace
    cd ..
}


while true; do
    read -p "Do you run the anchor_optimization without installing conda dependencies and building cython files? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) build_package; break;;
        * ) echo "Please answer yes or no.";;
    esac
done


python3 ./anchor_optimization/anchor_optimization/optimize_anchors_argparse.py --no-resize --include-stride --ratios=$RATIOS --scales=$SCALES --image-min-side=$IMAGE_MIN_SIDE --image-max-side=$IMAGE_MAX_SIDE $ANNOTATION_PATH
