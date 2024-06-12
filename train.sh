## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base.yaml \
   --name base_mse_quant --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device cuda:5 \
   > $filename 2>&1