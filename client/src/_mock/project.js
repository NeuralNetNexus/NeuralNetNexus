import { faker } from '@faker-js/faker';
import { sample } from 'lodash';

// ----------------------------------------------------------------------

const models = ['VGG-16', 'ResNet18', 'ResNet50', 'EfficiencyNet V2', 'AlexNet', 'ViT'];
const states = ['Initialized', 'Training', 'Completed', 'Failed'];

const projects = [...Array(24)].map((_, index) => ({
  id: faker.datatype.uuid(),
  name: `Project ${index + 1}`,
  dataset: `dataset${index + 1}.zip`,
  size: faker.datatype.number({ min: 1000, max: 5000 }),
  model: sample(models),
  state: sample(states)
}));

export default projects;
