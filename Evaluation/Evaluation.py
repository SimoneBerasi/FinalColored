from Evaluation import evaluate_experiment

anomaly_maps_dir = ''
dataset_base_dir = ''
output_dir = ''
pro_integration_limit = 0.3
evaluated_objects = 'carpet'

def evaluation():
    args = anomaly_maps_dir, dataset_base_dir, output_dir, pro_integration_limit, [evaluated_objects]
    evaluate_experiment.evaluate(args)



