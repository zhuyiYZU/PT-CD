source /home/mjy/anaconda3/bin/activate P_clickbait

do
  python fewshot.py --result_file ./output_fewshot.txt --dataset clickbait --template_id 0 --seed 123 --shot 10 --verbalizer manual
done