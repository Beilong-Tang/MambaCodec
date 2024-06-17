## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base_selm.yaml \
   --name dac_selm --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device 4,5,6,7 \
   > $filename 2>&1