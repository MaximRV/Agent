import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


class BankDepositModel:
    def __init__(self):
        self.data = {
            'Название банка': [
                'Банк А', 'Банк Б', 'Банк В', 'Банк Г',
                'Банк Д', 'Банк Е', 'Банк Ж', 'Банк З',
                'Банк И', 'Банк К', 'Банк Л', 'Банк М'
            ],
            'Местоположение': [
                'Москва', 'Санкт-Петербург', 'Екатеринбург', 'Новосибирск',
                'Казань', 'Нижний Новгород', 'Челябинск', 'Самара',
                'Уфа', 'Ростов-на-Дону', 'Волгоград', 'Красноярск'
            ],
            'Тип вклада': [
                'Срочный', 'Накопительный', 'Срочный', 'Накопительный',
                'Срочный', 'Накопительный', 'Срочный', 'Накопительный',
                'Срочный', 'Накопительный', 'Срочный', 'Накопительный'
            ],
            'Процентная ставка (%)': [
                7.5, 5.0, 6.0, 4.5,
                8.0, 5.5, 6.5, 4.0,
                7.0, 5.8, 6.2, 4.3
            ],
            'Минимальная сумма (руб.)': [
                10000, 5000, 20000, 1000,
                15000, 3000, 25000, 7000,
                12000, 4000, 18000, 2000
            ],
            'Срок вклада (мес.)': [
                12, 6, 24, 3,
                18, 12, 36, 6,
                24, 9, 15, 3
            ],
        }

        self.prepared_data = pd.DataFrame(self.data)
        self.model = self.train_model()

    def train_model(self):
        X = self.prepared_data[['Срок вклада (мес.)']]
        y = self.prepared_data['Процентная ставка (%)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X_train, y_train)
        return model

    def predict_rate(self, term):
        input_data = pd.DataFrame([[term]], columns=['Срок вклада (мес.)'])
        return self.model.predict(input_data)[0]

    def get_recommendations(self, term):
        return self.prepared_data[self.prepared_data['Срок вклада (мес.)'] == term].to_dict(orient='records')
