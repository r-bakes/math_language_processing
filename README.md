# Math Language Processing
My Penn State Schreyer undergraduate thesis work. Involves supplying attentional encoder decoder LSTMs and GRUs with elementary math problems and testing problem solving capabilities. 


The ultimate application of doing so was to develop education assistive software tools. The output papers propose the application of 
these models for the automatic generation of multiple choice question distractors given a sequence of provided test questions.

## Getting Started
This code was designed to be quickly and efficiently executable via command line to 
support rapid experimentation with model variables.

Using math_language_processing/main.py I would execute either a random forest, a GRU, 
or a now depricated LSTM on my schools on-prem server which
I could SSH onto via my own personal laptop.

## Style & Formatting
Code was written following PEP8 guidelines. Black was used to help enforce this. 
- [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines 
- [Black](https://black.readthedocs.io/en/stable/) code linter and formatter

## Related Publications
Output publications from this project:
- https://honors.libraries.psu.edu/catalog/6933rob5372
- Bakes, Riley Owen, et al.  “Educational Data Mining 2021.” Journal of Educational Data Mining, Math Question Solving and MCQ Distractor Generation with Attentional GRU Networks.


Paper which inspired the thesis:
- https://arxiv.org/abs/1904.01557

