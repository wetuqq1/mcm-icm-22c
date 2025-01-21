import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 数据预处理
def preprocess_data(btc_path, gold_path):
    date_format = '%m/%d/%y'
    btc = pd.read_csv(btc_path, parse_dates=['Date'], date_format=date_format)
    gold = pd.read_csv(gold_path, parse_dates=['Date'], date_format=date_format)
    
    btc.columns = ['Date', 'BTC']
    gold.columns = ['Date', 'GOLD']
    
    df = pd.merge(btc, gold, on='Date', how='outer').sort_values('Date')
    df = df.ffill().dropna()
    
    df['BTC_RET'] = np.log(df['BTC'] / df['BTC'].shift(1))
    df['GOLD_RET'] = np.log(df['GOLD'] / df['GOLD'].shift(1))
    
    return df.dropna()

# PyTorch数据集
class PriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = torch.FloatTensor(data).view(-1, 1)
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length], self.data[idx+self.seq_length]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# 模型训练
def train_model(asset, data, seq_length=30, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[[asset]])
    
    dataset = PriceDataset(scaled_data, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{asset}_model.pth')
            
        print(f'{asset} Epoch {epoch+1} | Loss: {avg_loss:.4f}')
    
    return model

class RiskParity:
    def __init__(self, window=21, max_cash_weight=0.5, cash_vol=0.3):
        self.window = window
        self.min_vol = 1e-6
        self.max_cash_weight = max_cash_weight  
        self.cash_vol = cash_vol  
        
    def get_weights(self, returns):
        all_returns = returns.copy()
        if 'CASH_RET' not in all_returns.columns:
            all_returns['CASH_RET'] = 0
        
      
        vol = all_returns.rolling(self.window, min_periods=1).std()
        vol = vol.replace(0, self.min_vol).fillna(self.min_vol)

        vol['CASH_RET'] = self.cash_vol
        
        inv_vol = 1 / vol
        # 确保权重和为1
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        # 限制现金权重
        weights['CASH_RET'] = weights['CASH_RET'].clip(upper=self.max_cash_weight)
        weights = weights.div(weights.sum(axis=1), axis=0)
        
        return weights

# 回测
class Backtester:
    def __init__(self, data, models):
        self.data = data 
        self.models = models  
        self.strategy = RiskParity()  
        self.fees = {'BTC': 0.02, 'GOLD': 0.01, 'CASH': 0.0} 
        
    def run(self):
        preds = []
        for asset in ['BTC', 'GOLD']:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.data[[asset]])
            
            model_preds = []
            self.models[asset].eval()
            with torch.no_grad():
                for i in range(30, len(scaled_data)):
                    seq = torch.FloatTensor(scaled_data[i-30:i]).view(1, -1, 1)
                    pred = self.models[asset](seq).numpy()[0][0]
                    model_preds.append(scaler.inverse_transform([[pred]])[0][0])
            
            preds.append(pd.Series(model_preds, index=self.data.index[30:], name=f'{asset}_PRED'))
        df = self.data.join(pd.concat(preds, axis=1)).dropna()
        # 添加现金收益率==0
        df['CASH_RET'] = 0.0
        
        portfolio = []
        prev_weights = None
        print(df.head())  # 调试输出
        for i in range(len(df)):
            if i < 51:  
                portfolio.append(1.0)
                if i == 50:
                    returns = df[['BTC_RET', 'GOLD_RET', 'CASH_RET']].iloc[:51]
                    prev_weights = self.strategy.get_weights(returns).iloc[-1]
                    print("Initial weights:", prev_weights)  
                continue
                
            returns = df[['BTC_RET', 'GOLD_RET', 'CASH_RET']].iloc[:i]
            weights = self.strategy.get_weights(returns).iloc[-1]
            # print(weights)

            asset_cols = ['BTC_RET', 'GOLD_RET', 'CASH_RET']
            current_returns = df[asset_cols].iloc[i]
            
            # 计算交易成本
            trading_cost = 0
            if prev_weights is not None:
                weight_changes = abs(weights - prev_weights)
                trading_cost = sum(weight_changes[col] * self.fees[col.split('_')[0]] 
                                 for col in asset_cols)
            
            ret = np.dot(weights[asset_cols], current_returns)
            portfolio.append(portfolio[-1] * np.exp(ret) * (1 - trading_cost))
            prev_weights = weights
            
        return df, portfolio

# 可视化
def plot_results(df, portfolio):
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3,1,1)
    plt.plot(df['Date'], df['BTC'], label='BTC Actual')
    plt.plot(df['Date'], df['BTC_PRED'], label='BTC Predicted', ls='--')
    plt.title('Bitcoin Price Predictions')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(df['Date'], df['GOLD'], label='Gold Actual')
    plt.plot(df['Date'], df['GOLD_PRED'], label='Gold Predicted', ls='--')
    plt.title('Gold Price Predictions')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(df['Date'][50:], portfolio[50:])
    plt.title('Portfolio Value')
    plt.xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

if __name__ == "__main__":
    df = preprocess_data('bitcoin.csv', 'gold.csv')
    
    # 训练并加载模型
    models = {}
    for asset in ['BTC', 'GOLD']:
        train_model(asset, df)
        model = LSTMModel()
        model.load_state_dict(torch.load(f'{asset}_model.pth', map_location='cpu'))
        models[asset] = model
    
    backtester = Backtester(df, models)
    result_df, portfolio = backtester.run()
    
    # 显示
    plot_results(result_df, portfolio)
    final_return = (portfolio[-1] - 1) * 100
    print(f'Final Portfolio Return: {final_return:.2f}%')