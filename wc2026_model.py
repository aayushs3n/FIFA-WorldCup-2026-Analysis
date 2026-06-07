
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
np.random.seed(42)
N_SIMS = 100000

teams_data = [
    {"team":"France","group":"B","elo":2002,"fifa_rank":3,"squad_val":1530,"form":9.5,"depth":10,"host":0},
    {"team":"Spain","group":"H","elo":1989,"fifa_rank":1,"squad_val":1260,"form":9.5,"depth":10,"host":0},
    {"team":"England","group":"L","elo":1975,"fifa_rank":4,"squad_val":1350,"form":8.5,"depth":9,"host":0},
    {"team":"Argentina","group":"G","elo":1970,"fifa_rank":2,"squad_val":900,"form":8.5,"depth":8,"host":0},
    {"team":"Brazil","group":"E","elo":1960,"fifa_rank":5,"squad_val":1140,"form":8.0,"depth":9,"host":0},
    {"team":"Portugal","group":"J","elo":1942,"fifa_rank":6,"squad_val":1020,"form":7.5,"depth":8,"host":0},
    {"team":"Germany","group":"A","elo":1938,"fifa_rank":9,"squad_val":1000,"form":7.5,"depth":8,"host":0},
    {"team":"Netherlands","group":"C","elo":1930,"fifa_rank":7,"squad_val":850,"form":7.0,"depth":8,"host":0},
    {"team":"Morocco","group":"F","elo":1895,"fifa_rank":11,"squad_val":568,"form":8.0,"depth":7,"host":0},
    {"team":"USA","group":"D","elo":1855,"fifa_rank":14,"squad_val":443,"form":7.0,"depth":7,"host":1},
    {"team":"Belgium","group":"K","elo":1890,"fifa_rank":8,"squad_val":750,"form":5.5,"depth":6,"host":0},
    {"team":"Colombia","group":"J","elo":1865,"fifa_rank":13,"squad_val":550,"form":7.0,"depth":7,"host":0},
    {"team":"Japan","group":"C","elo":1858,"fifa_rank":18,"squad_val":350,"form":7.5,"depth":7,"host":0},
    {"team":"Mexico","group":"A","elo":1845,"fifa_rank":15,"squad_val":380,"form":6.0,"depth":6,"host":1},
    {"team":"Uruguay","group":"H","elo":1852,"fifa_rank":16,"squad_val":300,"form":5.5,"depth":6,"host":0},
    {"team":"Senegal","group":"F","elo":1842,"fifa_rank":19,"squad_val":550,"form":7.0,"depth":6,"host":0},
    {"team":"Italy","group":"K","elo":1838,"fifa_rank":12,"squad_val":600,"form":6.0,"depth":6,"host":0},
    {"team":"Switzerland","group":"G","elo":1830,"fifa_rank":17,"squad_val":290,"form":6.5,"depth":6,"host":0},
    {"team":"Denmark","group":"B","elo":1825,"fifa_rank":21,"squad_val":280,"form":6.5,"depth":6,"host":0},
    {"team":"Croatia","group":"L","elo":1878,"fifa_rank":10,"squad_val":450,"form":6.0,"depth":6,"host":0},
    {"team":"South Korea","group":"I","elo":1812,"fifa_rank":22,"squad_val":280,"form":6.0,"depth":6,"host":0},
    {"team":"Austria","group":"E","elo":1808,"fifa_rank":24,"squad_val":320,"form":7.0,"depth":6,"host":0},
    {"team":"Turkey","group":"D","elo":1805,"fifa_rank":25,"squad_val":550,"form":6.5,"depth":6,"host":0},
    {"team":"Ecuador","group":"B","elo":1800,"fifa_rank":23,"squad_val":220,"form":6.0,"depth":5,"host":0},
    {"team":"Canada","group":"F","elo":1798,"fifa_rank":27,"squad_val":280,"form":6.5,"depth":6,"host":1},
    {"team":"Australia","group":"D","elo":1792,"fifa_rank":26,"squad_val":200,"form":5.5,"depth":5,"host":0},
    {"team":"Nigeria","group":"C","elo":1762,"fifa_rank":38,"squad_val":230,"form":5.5,"depth":5,"host":0},
    {"team":"Senegal","group":"F","elo":1842,"fifa_rank":19,"squad_val":550,"form":7.0,"depth":6,"host":0},
    {"team":"Morocco","group":"F","elo":1895,"fifa_rank":11,"squad_val":568,"form":8.0,"depth":7,"host":0},
    {"team":"Saudi Arabia","group":"B","elo":1728,"fifa_rank":60,"squad_val":100,"form":4.5,"depth":4,"host":0},
    {"team":"Panama","group":"A","elo":1705,"fifa_rank":30,"squad_val":80,"form":4.5,"depth":4,"host":0},
    {"team":"South Africa","group":"A","elo":1698,"fifa_rank":61,"squad_val":80,"form":4.0,"depth":4,"host":0},
]

df = pd.DataFrame(teams_data).drop_duplicates(subset="team").reset_index(drop=True)
scaler = MinMaxScaler()
df[["sq_s","form_s","dep_s"]] = scaler.fit_transform(df[["squad_val","form","depth"]])
df["rank_s"] = 1/df["fifa_rank"]
df["rank_s"] = (df["rank_s"]-df["rank_s"].min())/(df["rank_s"].max()-df["rank_s"].min())
df["comp_elo"] = df["elo"]+df["sq_s"]*60+df["form_s"]*80+df["dep_s"]*40+df["rank_s"]*50+df["host"]*50

groups={}
for _,r in df.iterrows():
    if r["group"] not in groups: groups[r["group"]]=[]
    groups[r["group"]].append(r["team"])
elo_map=dict(zip(df["team"],df["comp_elo"]))

def sim_match(ea,eb,ko=False):
    la=1.35*np.exp((ea-eb)/600); lb=1.35*np.exp((eb-ea)/600)
    ga,gb=np.random.poisson(la),np.random.poisson(lb)
    if ko and ga==gb:
        ga+=np.random.poisson(0.25*np.exp((ea-eb)/600))
        gb+=np.random.poisson(0.25*np.exp((eb-ea)/600))
        if ga==gb:
            ga+=1 if np.random.random()<np.clip(0.5+(ea-eb)/10000,0.35,0.65) else 0
            if ga==gb: gb+=1
    return ga,gb

def sim_groups():
    all_third,qualified=[],[]
    for g,teams in groups.items():
        st={t:{"pts":0,"gd":0,"gf":0} for t in teams}
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                t1,t2=teams[i],teams[j]; g1,g2=sim_match(elo_map[t1],elo_map[t2])
                st[t1]["gf"]+=g1; st[t2]["gf"]+=g2
                st[t1]["gd"]+=g1-g2; st[t2]["gd"]+=g2-g1
                if g1>g2: st[t1]["pts"]+=3
                elif g2>g1: st[t2]["pts"]+=3
                else: st[t1]["pts"]+=1; st[t2]["pts"]+=1
        s=sorted(st,key=lambda t:(st[t]["pts"],st[t]["gd"],st[t]["gf"]),reverse=True)
        qualified+=[s[0],s[1]]
        all_third.append({"team":s[2],"pts":st[s[2]]["pts"],"gd":st[s[2]]["gd"],"gf":st[s[2]]["gf"]})
    t3=pd.DataFrame(all_third).sort_values(["pts","gd","gf"],ascending=False).head(8)
    return qualified+t3["team"].tolist()

def sim_knockout(t32):
    prog={t:"R32" for t in t32}; bracket=t32[:]; np.random.shuffle(bracket); current=bracket
    for rnd in ["R16","QF","SF","Final","Winner"]:
        nxt=[]
        for i in range(0,len(current),2):
            if i+1>=len(current): nxt.append(current[i]); continue
            t1,t2=current[i],current[i+1]; g1,g2=sim_match(elo_map[t1],elo_map[t2],ko=True)
            w=t1 if g1>g2 else t2; l=t2 if g1>g2 else t1
            prog[l]=rnd; nxt.append(w)
        current=nxt
    if current: prog[current[0]]="Winner"
    return prog

stage_order=["Group","R32","R16","QF","SF","Final","Winner"]
results={t:{s:0 for s in stage_order} for t in df["team"]}
finals_tracker={}

print(f"Running {N_SIMS:,} simulations...")
for _ in range(N_SIMS):
    q32=sim_groups(); prog=sim_knockout(q32)
    all_t=[t for g in groups.values() for t in g]
    for t in all_t:
        if t not in prog: prog[t]="Group"
    for t,s in prog.items():
        if t in results: results[t][s]+=1
    fins=[t for t,s in prog.items() if s in ["Final","Winner"]]
    if len(fins)>=2:
        key=tuple(sorted(fins[:2])); finals_tracker[key]=finals_tracker.get(key,0)+1

rows=[]
for team in df["team"]:
    r=results[team]; T=N_SIMS
    rows.append({"Team":team,"Win%":round(r["Winner"]/T*100,2),
        "Final%":round((r["Final"]+r["Winner"])/T*100,2),
        "Semifinal%":round((r["SF"]+r["Final"]+r["Winner"])/T*100,2),
        "QF%":round((r["QF"]+r["SF"]+r["Final"]+r["Winner"])/T*100,2),
        "R16%":round((r["R16"]+r["QF"]+r["SF"]+r["Final"]+r["Winner"])/T*100,2)})

res=pd.DataFrame(rows).sort_values("Win%",ascending=False).reset_index(drop=True)
res.index+=1
print(res.head(15).to_string())

top_finals=sorted(finals_tracker.items(),key=lambda x:x[1],reverse=True)[:3]
print("\nMost Likely Finals:")
for m,c in top_finals: print(f"  {m[0]} vs {m[1]}: {c/N_SIMS*100:.1f}%")

fig1=px.bar(res.head(12),x="Win%",y="Team",orientation="h",color="Win%",
    color_continuous_scale="RdYlGn",text="Win%",
    title="🏆 2026 World Cup Win Probability — 100,000 Monte Carlo Simulations")
fig1.update_traces(texttemplate="%{text}%",textposition="outside")
fig1.update_layout(plot_bgcolor="#0d0d0d",paper_bgcolor="#0d0d0d",font_color="white",
    yaxis={"categoryorder":"total ascending"},showlegend=False,height=550)
fig1.show()
fig1.write_html("chart_win_probability.html")

heatmap_cols=["R16%","QF%","Semifinal%","Final%","Win%"]
fig2=px.imshow(res.head(12).set_index("Team")[heatmap_cols],
    color_continuous_scale="YlOrRd",text_auto=True,
    title="📊 Tournament Progression Heatmap")
fig2.update_layout(plot_bgcolor="#0d0d0d",paper_bgcolor="#0d0d0d",font_color="white",height=500)
fig2.show()
fig2.write_html("chart_heatmap.html")

print("\n✅ Done! Charts saved as HTML files in your FIFA_Project folder!")
print("Open them in Chrome and screenshot with Cmd+Shift+4")
