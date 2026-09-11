from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'gamma_out')
OUT.mkdir(parents=True, exist_ok=True)
DATA = sys.argv[1]
YEARS = [2023, 2024, 2025, 2026]
PRIOR = {'sig':3.8,'sigpct':0.46,'sub':0.45,'td':1.45,'tdpct':0.40}
RATE_SPECS = [
    ('avg_sig_str_landed','sig'),('avg_sig_str_pct','sigpct'),
    ('avg_sub_att','sub'),('avg_td_landed','td'),('avg_td_pct','tdpct')]
COUNT_COLS = [
    'current_win_streak','current_lose_streak','longest_win_streak','wins','losses',
    'total_rounds_fought','total_title_bouts','win_by_ko_tko','win_by_submission',
    'height_cms','reach_cms','age']


def implied(a):
    a=np.asarray(a,dtype=float)
    return np.where(a<0, -a/(-a+100.0), 100.0/(a+100.0))

def dec(a):
    a=np.asarray(a,dtype=float)
    return np.where(a<0, 1+100.0/(-a), 1+a/100.0)

def logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-40,40)
    return 1/(1+np.exp(-x))

def side_num(df, side, stem):
    return pd.to_numeric(df[f'{side}_{stem}'], errors='coerce')

def shrink_rate(df, side, stem, prior):
    rounds=side_num(df,side,'total_rounds_fought').fillna(0).clip(lower=0)
    w=rounds/(rounds+6.0)
    v=side_num(df,side,stem)
    return w*v + (1-w)*prior

def features(df):
    x={}
    for c in COUNT_COLS:
        rv=side_num(df,'r',c); bv=side_num(df,'b',c)
        x[c+'_diff']=rv-bv
    for stem,pkey in RATE_SPECS:
        x[stem+'_shr_diff']=shrink_rate(df,'r',stem,PRIOR[pkey])-shrink_rate(df,'b',stem,PRIOR[pkey])
    rr=pd.to_numeric(df.get('r_match_weightclass_rank'),errors='coerce').fillna(25)
    br=pd.to_numeric(df.get('b_match_weightclass_rank'),errors='coerce').fillna(25)
    x['rank_adv']=br-rr
    title=pd.Series(df.get('title_bout',False),index=df.index).astype(str).str.upper().eq('TRUE').astype(float)
    rounds=pd.to_numeric(df.get('no_of_rounds',3),errors='coerce').fillna(3)
    x['win_diff_x_title']=(side_num(df,'r','wins')-side_num(df,'b','wins'))*title
    x['round_diff_x_5r']=(side_num(df,'r','total_rounds_fought')-side_num(df,'b','total_rounds_fought'))*(rounds>=5).astype(float)
    return pd.DataFrame(x,index=df.index)

def valid_rows(raw):
    z=raw.copy(); z['date']=pd.to_datetime(z['date'],errors='coerce')
    z=z[z['winner'].isin(['Red','Blue'])].copy()
    z['r_odds']=pd.to_numeric(z['r_odds'],errors='coerce'); z['b_odds']=pd.to_numeric(z['b_odds'],errors='coerce')
    z=z[z['date'].notna() & z['r_odds'].notna() & z['b_odds'].notna()].copy()
    ri=implied(z.r_odds); bi=implied(z.b_odds); s=ri+bi
    z['pm_r']=ri/s; z['y']=(z.winner=='Red').astype(int)
    return z.sort_values('date').reset_index(drop=True)

def model():
    return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(C=.35,solver='lbfgs',max_iter=3000))])

def fit_fund(X,y):
    Xa=pd.concat([X,-X],ignore_index=True); ya=np.concatenate([y,1-y])
    m=model(); m.fit(Xa,ya); return m

def probs(m,X): return m.predict_proba(X)[:,1]

def metric(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return {'accuracy':float(accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p,labels=[0,1]))}

def split_event_dates(z, frac=.80):
    d=np.array(sorted(pd.to_datetime(z.date.dt.normalize().unique())))
    if len(d)<10: return None
    k=max(1,min(len(d)-2,int(len(d)*frac)))
    return pd.Timestamp(d[k])

def residual_params(pm,pf,y):
    x=logit(pf)-logit(pm); base=logit(pm)
    def loss(t): return log_loss(y,sigmoid(base+t[0]+t[1]*x),labels=[0,1])
    r=minimize(loss,[0.,0.],method='Nelder-Mead',options={'maxiter':2000})
    a,b=map(float,r.x); b=float(np.clip(b,-1.0,1.0)); a=float(np.clip(a,-.5,.5))
    return a,b

def grade_tier(g):
    n=len(g)
    if n==0:return {'n':0,'wins':0,'hit':None,'units':0.0,'roi':None}
    wins=int(g['won'].sum()); units=float(g['profit'].sum())
    return {'n':n,'wins':wins,'hit':wins/n,'units':units,'roi':units/n}

raw=pd.read_csv(DATA,low_memory=False)
z=valid_rows(raw)
Xall=features(z)
metrics=[]; ledger=[]; gates=[]
for yr in YEARS:
    tr=z[z.date < pd.Timestamp(f'{yr}-01-01')].copy()
    te=z[(z.date>=pd.Timestamp(f'{yr}-01-01')) & (z.date<pd.Timestamp(f'{yr+1}-01-01'))].copy()
    if tr.empty or te.empty: continue
    cut=split_event_dates(tr,.80)
    base=tr[tr.date<cut].copy(); cal=tr[tr.date>=cut].copy()
    cut2=split_event_dates(cal,.50)
    cfit=cal[cal.date<cut2].copy(); cval=cal[cal.date>=cut2].copy()
    mb=fit_fund(Xall.loc[base.index],base.y.values)
    pf_fit=probs(mb,Xall.loc[cfit.index]); pf_val=probs(mb,Xall.loc[cval.index])
    a,b=residual_params(cfit.pm_r.values,pf_fit,cfit.y.values)
    pval=sigmoid(logit(cval.pm_r.values)+a+b*(logit(pf_val)-logit(cval.pm_r.values)))
    mm=metric(cval.y,cval.pm_r); mr=metric(cval.y,pval)
    accepted=(mr['brier']<mm['brier'] and mr['logloss']<mm['logloss'])
    if not accepted: a,b=0.,0.
    mf=fit_fund(Xall.loc[tr.index],tr.y.values)
    pf=probs(mf,Xall.loc[te.index]); pm=te.pm_r.values
    final=sigmoid(logit(pm)+a+b*(logit(pf)-logit(pm)))
    for name,p in [('market',pm),('fund',pf),('residual',final)]:
        metrics.append({'year':yr,'model':name,'n':len(te),**metric(te.y.values,p),'residual_accepted':accepted,'a':a,'beta':b})
    gates.append({'year':yr,'accepted':accepted,'cal_n':len(cal),'fit_n':len(cfit),'val_n':len(cval),'market_brier':mm['brier'],'resid_brier':mr['brier'],'market_logloss':mm['logloss'],'resid_logloss':mr['logloss'],'a':a,'beta':b})
    for j,(idx,row) in enumerate(te.iterrows()):
        p=float(final[j]); f=float(pf[j]); m=float(pm[j]); y=int(row.y)
        red=p>=.5; ps=p if red else 1-p; fs=f if red else 1-f; ms=m if red else 1-m
        odd=float(row.r_odds if red else row.b_odds); dd=float(dec([odd])[0]); won=int((y==1)==red)
        ev=ps*dd-1; edge=ps-ms
        tier='PASS'
        if ev>=.04 and edge>=.025 and ps>=.57 and 1.20<=dd<=3.50: tier='B'
        if ev>=.08 and edge>=.05 and ps>=.62 and 1.25<=dd<=3.50: tier='C'
        if dd>=1.40 and ps>=.76 and fs>=.70 and edge>=.06 and ev>=.08: tier='TITAN_CANDIDATE'
        ledger.append({'year':yr,'date':row.date.date().isoformat(),'red':row.r_fighter,'blue':row.b_fighter,'pick':row.r_fighter if red else row.b_fighter,'american':odd,'decimal':dd,'pm_side':ms,'pf_side':fs,'p_side':ps,'edge':edge,'ev':ev,'tier':tier,'won':won,'profit':(dd-1 if won else -1),'residual_accepted':accepted})

M=pd.DataFrame(metrics); L=pd.DataFrame(ledger); G=pd.DataFrame(gates)
M.to_csv(OUT/'metrics.csv',index=False); L.to_csv(OUT/'ledger.csv',index=False); G.to_csv(OUT/'residual_gates.csv',index=False)
rows=[]
for (year,tier),g in L[L.tier!='PASS'].groupby(['year','tier']): rows.append({'year':year,'tier':tier,**grade_tier(g)})
for tier,g in L[L.tier!='PASS'].groupby('tier'): rows.append({'year':'ALL','tier':tier,**grade_tier(g)})
T=pd.DataFrame(rows); T.to_csv(OUT/'tiers.csv',index=False)
fav=np.where(z.pm_r.values>=.5,z.y.values,1-z.y.values)
summary={'source_rows':int(len(raw)),'usable_rows':int(len(z)),'market_favorite_accuracy_all':float(np.mean(fav)),'titan':grade_tier(L[L.tier=='TITAN_CANDIDATE']),'policy':'gamma diagnostic fixed before results; does not modify UFCC1 alpha/beta'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('=== METRICS ==='); print(M.to_string(index=False))
print('\n=== RESIDUAL GATES ==='); print(G.to_string(index=False))
print('\n=== TIERS ==='); print(T.to_string(index=False) if not T.empty else 'none')
print('\n=== SUMMARY ==='); print(json.dumps(summary,indent=2))
