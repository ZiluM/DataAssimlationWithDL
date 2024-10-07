module load conda
conda activate earth2mip

### Debugging

nvidia-smi
module list
python -c "import torch; print(torch.cuda.is_available())"


### Run the model

cd /glade/work/zilumeng/3D_trans

python ./da.py ./config.yml

echo "Done, sub post da"



