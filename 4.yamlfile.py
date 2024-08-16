import yaml

data = {'train' : './train/images',
               'val' : './valid/images',
               'test' : './test/images',
               'names' : ['person'],
               'nc' : 1 }

with open('./person.yaml','w') as f:
  yaml.dump(data,f)

with open('./person.yaml', 'r') as f:
  data_yaml = yaml.safe_load(f)
  print(data_yaml)