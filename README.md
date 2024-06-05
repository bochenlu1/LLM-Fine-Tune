**Improving Financial Sentiment Analysis with Fine-Tuned BERT Models: A Comparative Study of Pre-Trained Corpora**

Sentiment analysis has become increasingly important in the financial domain with a wide range of applications (Fatouros et al., 2023) (Remesh & Gaba, 2023). Accurate sentiment analysis of financial news and reports can provide valuable insights for investors, traders, and financial institutions, enabling them to make informed decisions and manage risk (Remesh & Gaba, 2023). However, the complexity and specialized (domain-specific) language used in financial texts pose unique challenges when running these models. As a result, we wanted to compare the performance of three different BERT-based language models (BERT-Large-uncased, BERT-uncased, and FinBERT) on financial sentiment analysis tasks. By fine-tuning these models and optimizing hyperparameters such as number of epochs and learning rate, we aim to:

●	Investigate the impact of different pre-training corpora on the models’ performance.

●	Improve the accuracy of financial sentiment analysis by fine-tuning.

●	Identify the optimal model and hyperparameter settings to achieve the best performance (accuracy
).

The uniqueness of our approach is in the comparative analysis of the various BERT-based language models for financial sentiment analysis. While there has been comparative analysis across companies (i.e., ChatGPT v. BERT), we have chosen to look into BERT models to provide insights into how the choice of pre-trained corpus affects the models’ performance and whether fine–tuning can bridge any gaps (Fatouros et al., 2023). By identifying the best-performing model, we can enable more accurate and reliable sentiment analysis of financial text.

**Methods & Code Description:**

a.	Setup 

Class code 5 was taken and adapted to suit the appropriate finance dataset. All testing was done on Google Colab and Kaggle Notebooks.

b.	Dataset 

HuggingFace’s financial phrasebank (curated by Takala) dataset was used for our experiments. It is a collection of 4,840 sentences from financial news articles that are in English. The sentences were annotated by humans and categorized by sentiment as either positive, negative, or neutral. Multiple annotators review each sentence, allowing for the creation of different data splits based on the level of agreement among the annotators. The dataset is provided in four different versions – sentences_50agree (with at least 50% agreement), sentences _66agree (at least 66% agreement), sentences _75agree (at least 75% agreement), and sentences _allagree (100% agreement) (Malo & Sinha, 2022).

c.	Models Used

Three models were used to conduct our testing: BERT-Large-uncased, BERT-uncased, and FinBERT.

d.	Baseline Testing

Models were assessed without any form of training on 10% of randomly sampled validation data. The following table contains the mean accuracy score of the tested models over 30 iterations. A summary of baseline testing results in table 1 below:

e.	Fine-Tuning

Considering limited data and computational resources, we leveraged a transfer-learning fine-tuning technique by feeding the 3 pre-trained BERT models data specific to sentiment classification of financial news. Various “agree” categories of the financial_phrase Hugging Face dataset were used, as well as hyper-parameter tuning such as adjusting epochs, learning rates, batch sizes to optimize the training process.

To conserve time and effort, a BERT-uncased model was used to perform hyperparameter tuning as it presents the least amount of parameters (110M). The hyperparameters which returned the highest accuracy were then applied across BERT-Large-uncased (320M), FinBERT.

Training layers & hyper-parameters:

In the training loop, the input data is passed through the BERT model, consisting of embedding, transformer, and classification layers, to process tokens and generate contextual representations. The optimizer, an external component, updates the model's parameters based on computed gradients during backpropagation, facilitating performance improvement. Utilizing the CrossEntropyLoss function, the difference between model predictions and actual labels is quantified to guide parameter adjustments. Through backpropagation, gradients are computed and used by the optimizer to refine the model's weights, ultimately enhancing its accuracy in sentiment analysis tasks.

After some experimentation, we finalized on the following hyperparameters as it gave us a good balance between training time and loss.

●	Learning rate: 1e-5

●	Optimizer: AdamW

●	Batch Size: 32

●	Number of Training Epochs: 30
