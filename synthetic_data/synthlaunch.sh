export TRANSFORMERS_VERBOSITY=error
for ((i=0; i<8; i++)); do
  CUDA_VISIBLE_DEVICES=$i py synth_data_runner.py --gpu $i &
done