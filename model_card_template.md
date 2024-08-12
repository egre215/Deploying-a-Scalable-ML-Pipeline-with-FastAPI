# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used is LogisticRegression() using FastAPI.

## Intended Use

This is intended for a project in Udacity (Machine Learning DevOPS).

## Training Data

The data is trained to determine whether or not an adult's income is greater than or equal to 50 thousand dollars.

## Evaluation Data

The testing data is used to determine the efficiency of the model in predicting a person's income.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The metrics used were precision, recall, f1. The performance of these metrics can be found in the "slice_output.txt" file.

## Ethical Considerations

This data should not be used to reverse engineer personal information of the individuals involved. Furthermore, any extraction of person information should be prohibited.

## Caveats and Recommendations

This is only intended for a Udacity project and should not be used to acurately predict a person's income.