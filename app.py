import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== 데이터 불러오기 및 전처리 =====
df = pd.read_csv("data/Health_Sleep_Statistics.csv")

df = df.drop(columns=['User ID'])  # 불필요한 컬럼 제거

# 결측치 처리 (NaN 값 제거)
df = df.dropna()

# 범주형 변수 인코딩
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_physical_activity_level = LabelEncoder()
df['Physical Activity Level'] = le_physical_activity_level.fit_transform(df['Physical Activity Level'])

le_dietary_habits = LabelEncoder()  
df['Dietary Habits'] = le_dietary_habits.fit_transform(df['Dietary Habits']) 

le_sleep_disorders = LabelEncoder()
df['Sleep Disorders'] = le_sleep_disorders.fit_transform(df['Sleep Disorders'])

df['Medication Usage'] = df['Medication Usage'].map({'no': 0, 'yes': 1})

# Sleep Quality를 인코딩 (타겟 변수)
le_sleep_quality = LabelEncoder()
df['Sleep Quality'] = le_sleep_quality.fit_transform(df['Sleep Quality'])

# 'Sleep Duration' 계산
def convert_time(bedtime, wakeup):
    bedtime = pd.to_datetime(bedtime, format='%H:%M')
    wakeup = pd.to_datetime(wakeup, format='%H:%M')
    sleep_duration = (wakeup - bedtime).dt.total_seconds() / 3600
    sleep_duration[sleep_duration < 0] += 24  # 음수 값 보정
    return sleep_duration

df['Sleep Duration'] = convert_time(df['Bedtime'], df['Wake-up Time'])
df = df.drop(columns=['Bedtime', 'Wake-up Time'])

# print(df.dtypes)

# 특성과 타겟 분리
X = df.drop(columns=['Sleep Quality'])
y = df['Sleep Quality']

# 범주형 변수(예: 'Age', 'Daily Steps' 등)가 문자열로 되어 있으면 변환을 해야합니다. 예를 들어, 'medium', 'low'와 같은 값이 있다면 이를 처리해야 합니다.

# 예를 들어, 'Daily Steps'에 'medium', 'low'와 같은 값이 있다면 이를 LabelEncoder로 인코딩합니다.
# 아래는 수치형으로 변환할 수 없는 데이터를 처리하는 코드 예시:

categorical_columns = ['Daily Steps', 'Calories Burned']  # 이곳에 범주형 변수가 있으면 추가

for col in categorical_columns:
    if df[col].dtype == 'object':  # 만약 컬럼에 문자열이 포함된 경우
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# ===== Streamlit UI 구성 =====
st.set_page_config(page_title='Health & Sleep Dashboard', layout='wide')
st.sidebar.title('Navigation')
menu = st.sidebar.radio('Go to', ['Home', 'EDA', 'Model Performance'])

def home():
    st.title('Health & Sleep Dashboard')
    st.markdown('''  
    - **Age**: 나이
    - **Gender**: 성별 (0: Female, 1: Male)
    - **Daily Steps**: 하루 걸음 수
    - **Calories Burned**: 소모 칼로리
    - **Sleep Disorders**: 수면 장애 여부 (0: No, 1: Yes)
    - **Medication Usage**: 약물 복용 여부 (0: No, 1: Yes)
    - **Sleep Duration**: 수면 시간 (시간 단위)
    - **Sleep Quality**: 수면의 질 (타겟 변수)
    ''')

def eda():
    st.title('데이터 시각화')
    chart_tabs = st.tabs(['Histogram', 'Boxplot', 'Heatmap'])
    
    with chart_tabs[0]:
        st.subheader('Feature Distributions')
        fig, axes = plt.subplots(2,3, figsize=(15,10))
        columns = ['Age', 'Daily Steps', 'Calories Burned', 'Sleep Duration']
        for i, col in enumerate(columns):
            ax = axes[i//3, i%3]
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(col)
        st.pyplot(fig)
    
    with chart_tabs[1]:
        st.subheader('Boxplot: Sleep Duration by Sleep Disorders')
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x='Sleep Disorders', y='Sleep Duration', palette='Set2', ax=ax)
        st.pyplot(fig)
    
    with chart_tabs[2]:
        st.subheader('Feature Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)














def model_performance():
    st.title('모델 성능 평가')
    st.write(f'**Accuracy:** {accuracy:.2f}')
    st.text('Classification Report:')
    st.text(classification_rep)

if menu == 'Home':
    home()
elif menu == 'EDA':
    eda()
elif menu == 'Model Performance':
    model_performance()
