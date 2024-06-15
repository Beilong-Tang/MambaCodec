## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base.yaml \
   --name dac_mse --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device 1,2,3 \
   > $filename 2>&1