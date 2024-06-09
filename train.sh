## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base_mse.yaml \
   --name base_mse --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device cuda:5 \
   --continue_from /DKUdata/tangbl/MambaTransformer/ckpt/base_mse/epoch3.pth \
   > $filename 2>&1