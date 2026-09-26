# Exact reproduction of the revision notebook (revise_slte.ipynb), cells 2-4.
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def load_real():
    df = pd.read_csv('MedicalAppointment.csv')
    df = df[df['Age'] != -1].copy()
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    df['ScheduledDayOfWeek'] = df['ScheduledDay'].dt.dayofweek
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek
    df = df.drop(columns=['ScheduledDay', 'AppointmentDay'])
    neigh_freq = df['Neighbourhood'].value_counts(normalize=True)
    df['Neighbourhood'] = df['Neighbourhood'].map(neigh_freq)
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
    df = df.drop(columns=['PatientId', 'AppointmentID'])
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['No-show']); y = df['No-show']
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

def load_synth(path, cols):
    s = pd.read_csv(path)
    if 'Unnamed: 0' in s.columns: s = s.drop(columns=['Unnamed: 0'])
    if 'Neighbourhood_freq' in s.columns: s = s.rename(columns={'Neighbourhood_freq': 'Neighbourhood'})
    if 'ScheduledMonth' in s.columns: s = s.drop(columns=['ScheduledMonth'])
    return s.drop(columns=['No-show'])[cols], s['No-show']

def compute_slte_once(X_train, y_train, synth_X, synth_y, scale=True, return_low=False):
    Xtr, Xsy = X_train.copy(), synth_X.copy()
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr = pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
        Xsy = pd.DataFrame(sc.transform(Xsy), columns=Xsy.columns, index=Xsy.index)
    anchor = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42).fit(Xtr, y_train)
    proba = anchor.predict_proba(Xsy); classes = list(anchor.classes_)
    lcs = np.array([proba[i, classes.index(synth_y.iloc[i])] for i in range(len(synth_y))])
    corrected = synth_y.values.copy(); low = lcs < 0.40
    if low.sum() > 0:
        knn = KNeighborsClassifier(n_neighbors=7).fit(Xtr, y_train)
        corrected[low] = knn.predict(Xsy[low])
    weights = np.where(lcs >= 0.70, 1.0, np.where(lcs >= 0.40, 0.7, 0.5))
    return pd.Series(corrected, index=synth_y.index), pd.Series(weights, index=synth_y.index), lcs

class ScaledLogisticRegression(LogisticRegression):
    """Logistic Regression on standardised inputs (scaler fitted on each training set). Used for the Adult data only,
    where lbfgs does not converge within max_iter = 1000 on the unscaled capital-gain column (0 to 99,999)."""
    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(X)
        return super().fit(self.scaler_.transform(X), y, sample_weight=sample_weight)

    def decision_function(self, X):   # predict and predict_proba go through decision_function
        return super().decision_function(self.scaler_.transform(X))


def make_classifiers(seed=42, scaled_lr=False):
    LR = ScaledLogisticRegression if scaled_lr else LogisticRegression
    return {
        'Logistic Regression': LR(max_iter=1000, class_weight='balanced', random_state=seed),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=seed),
    }
