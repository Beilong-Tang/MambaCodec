## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base_sisnr.yaml \
   --name base_sisnr --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device cuda:7 \
   > $filename 2>&1