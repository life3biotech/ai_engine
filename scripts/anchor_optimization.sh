IMAGE_MIN_SIDE=640
IMAGE_MAX_SIDE=640
SCALES=5
RATIOS=5
ANNOTATION_PATH='./data/processed/annotations_train_efficientdet.csv'

# Build package
build_package() 
{
    cd anchor_optimization
    python3 setup.py build_ext --inplace
    cd ..
}


while true; do
    read -p "Run anchor optimization tool without building Cython files? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) build_package; break;;
        * ) echo "Please answer yes (y) or no (n).";;
    esac
done


python3 ./src/anchor_optimization/anchor_optimization/optimize_anchors_argparse.py --no-resize --ratios=$RATIOS --scales=$SCALES --image-min-side=$IMAGE_MIN_SIDE --image-max-side=$IMAGE_MAX_SIDE $ANNOTATION_PATH
# Possible args: --include-stride