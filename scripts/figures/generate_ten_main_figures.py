"""Recompose the existing evidence into ten compact manuscript figures.

The original numerical panel builders are reused without fitting models. Each
panel is rendered independently as well as within its final combined figure.
"""
from pathlib import Path
import sys, json, importlib, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import figstyle as st

DEST=ROOT/'figures/main10'
QA=ROOT/'docs/figure_review/main10'
DEST.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
CURRENT=None
SOURCES={}
ORIGINS={'Fig1':'survey','Fig2':'transfer','Fig3':'representation','Fig4':'recalibration',
         'Fig5':'intervals_classification','Fig6':'ammonium_associations',
         'FigS1':'sequencing_replicates','FigS2':'country_offsets','FigS3':'stacking',
         'FigS4':'shrinkage_sensitivity','FigS5':'interval_sensitivity','FigS6':'free_ammonia_proxy',
         'FigS7':'primer_concordance','FigS8':'resolution_classification'}

def route_source(name,fname,df):
    if CURRENT is None:
        SOURCES.setdefault(name,{})[fname]=df.copy()
    else:
        out=DEST/CURRENT/'source_data';out.mkdir(parents=True,exist_ok=True)
        df.to_csv(out/f'{ORIGINS.get(name,name)}_{fname}',index=False)

def compact_hgrid(ss,ratios,gap=0.43):
    if CURRENT in ('Fig6','Fig7','Fig8') and len(ratios)>=4:
        return st.hgrid(ss,ratios,gap=max(gap,.60))
    if CURRENT=='Fig4' and len(ratios)==2 and gap>=.9:
        return st.hgrid(ss,ratios,gap=gap)
    return st.hgrid(ss,ratios,gap=min(gap,0.43))

def compact_vgrid(ss,ratios,gap=0.43):
    return st.vgrid(ss,ratios,gap=min(gap,0.43))

MODULES={n:importlib.import_module(n) for n in ['fig1_data_design','fig2_ceiling','fig3_error_source',
    'fig4_fewshot_eb','fig5_intervals_alarms','fig6_dml','figS_supplementary']}
for mod in MODULES.values():
    mod.source_data=route_source
    mod.hgrid=compact_hgrid
    if hasattr(mod,'vgrid'):mod.vgrid=compact_vgrid
f1,f2,f3,f4,f5,f6,fs=[MODULES[n] for n in MODULES]

CAPTURED={}
class Capture(st.Composite):
    def panel(self,letter,ss,draw):self.records.append({'key':letter,'draw':draw})
    def finish(self,*args,**kwargs):
        CAPTURED[self.name]={r['key']:r['draw'] for r in self.records}
        plt.close(self.fig)
        return []

def get_supp(n):
    name=f'FigS{n}'
    if name not in CAPTURED:
        save=fs.Composite;fs.Composite=Capture
        try:getattr(fs,f'fig_s{n}')()
        finally:fs.Composite=save
    return CAPTURED[name]

MAPPING={
 'Fig1':{'title':'Survey, community profiles and validation design','old':['1a','1b','1c','1d','S1a','S1b'],
         'stem':'Fig1_survey_design','height':11.6},
 'Fig2':{'title':'Cross-country prediction and country offsets','old':['2a','2b','2c','2d','2e','2f'],
         'stem':'Fig2_cross_country_prediction','height':13.0},
 'Fig3':{'title':'Model stacking across countries','old':['2g','S3a','S3b','S3c','S3d'],
         'stem':'Fig3_stacking','height':10.2},
 'Fig4':{'title':'Taxonomic resolution and geographic reliance','old':['3a','S8a','S8b','S8c','3c','3d','3e'],
         'stem':'Fig4_resolution_geographic_reliance','height':11.8},
 'Fig5':{'title':'Primer transfer and feature-level remedies','old':['3f','S7a','S7b','3g'],
         'stem':'Fig5_primer_feature_remedies','height':8.0},
 'Fig6':{'title':'Few-label country recalibration','old':['1e','4a','4b','4g','4h'],
         'stem':'Fig6_few_label_recalibration','height':11.8},
 'Fig7':{'title':'Shrinkage components and label budgets','old':['4c','4d','4e','4f','S4a','S4b'],
         'stem':'Fig7_shrinkage_label_budget','height':11.8},
 'Fig8':{'title':'Country-aware prediction intervals','old':['5a','5b','S5a','S5b','5c','S5c'],
         'stem':'Fig8_prediction_intervals','height':16.4},
 'Fig9':{'title':'Operating-state classification across countries','old':['5d','5e','5f','S8d','S8e','S8f'],
         'stem':'Fig9_operating_state_classification','height':8.7},
 'Fig10':{'title':'Within-country ammonium associations','old':['6a','6b','6c','6d','6e','6f','S6a','S6b'],
          'stem':'Fig10_ammonium_associations','height':15.8},
 'FigS1':{'title':'Country prediction offsets','old':['S2a','S2b','S2c','S2d','S2e'],
          'stem':'FigS1_country_offsets','height':9.2},
}
MAPPING['Fig5']['old']=['S7a','S7b','3g']
for name,d in MAPPING.items():d['new_panels']={old:chr(97+i) for i,old in enumerate(d['old'])}
MAP={'figures':MAPPING,'deleted_duplicate':{'3b':'4b','3f':'5b'},
     'notes':'Old 3b duplicates the LOCO lines retained within new 4b, which also includes GKF5. Old 3f duplicates four primer scenarios retained within the complete six-scenario new 5b. Old S2 is retained as new SI FigS1.'}
(DEST/'figure_map.json').write_text(json.dumps(MAP,indent=2),encoding='utf-8')

def builders(name):
    if name=='Fig1':
        d=f1.digesters();g=f1.map_frame(d);sp=get_supp(1)
        return [lambda F,S:f1.draw_a(F,S,g),lambda F,S:f1.draw_b(F,S,d),lambda F,S:f1.draw_c(F,S,d),f1.draw_d,sp['a'],sp['b']]
    if name=='Fig2':
        dec,offs=f2.decomposition()
        route_source('Fig2','error_decomposition.csv',dec)
        route_source('Fig2','country_offsets.csv',offs)
        return [f2.draw_a,f2.draw_b,lambda F,S:f2.draw_c(F,S,dec),lambda F,S:f2.draw_d(F,S,dec),lambda F,S:f2.draw_e(F,S,offs),f2.draw_g]
    if name=='Fig3':
        sp=get_supp(3);return [draw_stacking_overview]+[sp[x] for x in 'abcd']
    if name=='Fig4':
        sp=get_supp(8)
        def offset_resolution(fig,ss):
            before=set(fig.axes);sp['b'](fig,ss)
            for ax in [a for a in fig.axes if a not in before]:
                if ax.get_legend():
                    for text in ax.get_legend().get_texts():text.set_fontsize(10.5)
        return [f3.draw_a,sp['a'],offset_resolution,sp['c'],f3.draw_c,f3.draw_d,f3.draw_e]
    if name=='Fig5':
        sp=get_supp(7)
        def primer_combinations(fig,ss):
            before=set(fig.axes);sp['b'](fig,ss)
            for ax in [a for a in fig.axes if a not in before]:ax.set_ylabel('LOCO R$^2$')
        return [sp['a'],primer_combinations,f3.draw_g]
    if name=='Fig6':
        pc=pd.concat([f4.per_country(t) for t in f4.TG])
        route_source('Fig4','country_recalibration_k3.csv',pc)
        return [draw_recalibration_scheme,f4.draw_a,f4.draw_b,lambda F,S:f4.draw_country_mae(F,S,pc),lambda F,S:f4.draw_means(F,S,pc)]
    if name=='Fig7':
        sp=get_supp(4);return [f4.draw_ratio,f4.draw_weights,f4.draw_ablation,f4.draw_expected,sp['a'],sp['b']]
    if name=='Fig8':
        sp=get_supp(5);c=pd.read_csv(st.RES/'fewshot_conformal_LOCO.csv');s=f5.pooled(c[c.alpha==.1]);s=s[s.rule.isin(f5.R_COL)]
        return [lambda F,S:f5.draw_a(F,S,s),lambda F,S:f5.draw_b(F,S,s),sp['a'],sp['b'],lambda F,S:f5.draw_c(F,S,c),sp['c']]
    if name=='Fig9':
        sp=get_supp(8);return [f5.draw_d,f5.draw_e,f5.draw_f,sp['d'],sp['e'],sp['f']]
    if name=='Fig10':
        sp=get_supp(6);w=f6.load();r=pd.read_csv(st.RES/'dml_refutation_parallel.csv');d=f6.bw_table();s=f6.bw_slopes(d)
        return [lambda F,S:f6.draw_a(F,S,w,r),lambda F,S:f6.draw_b(F,S,w),lambda F,S:f6.draw_c(F,S,d,s),lambda F,S:f6.draw_d(F,S,s),lambda F,S:f6.draw_e(F,S,r),lambda F,S:f6.draw_f(F,S,r),sp['a'],sp['b']]
    if name=='FigS1':return list(get_supp(2).values())
    raise KeyError(name)

def draw_recalibration_scheme(fig,ss):
    """The exact shrinkage equations remain vector text over a pictorial context."""
    asset=DEST/'assets/recalibration_illustration.png'
    if not asset.exists():
        return f1.draw_e(fig,ss)
    grid=st.hgrid(ss,[3.05,1],gap=.18)
    ax=fig.add_subplot(grid[0]);ax.set_axis_off()
    image=np.asarray(Image.open(asset).convert('RGB'))
    ax.imshow(image,aspect='equal',zorder=0)
    labels=[(1/6,'Training countries'),(.5,'k local labels'),(5/6,'Country recalibration')]
    for x,text in labels:ax.text(x,-.045,text,transform=ax.transAxes,ha='center',va='top',fontsize=13,color=st.INK)
    right=fig.add_subplot(grid[1]);right.set_axis_off()
    right.text(.02,.92,'Residual variance',ha='left',va='top',fontsize=13,color=st.INK)
    right.text(.02,.81,r'$\hat\tau^2$ between;  $\hat\sigma^2$ within',ha='left',va='top',fontsize=13,color=st.INK)
    right.text(.02,.61,r'$\hat b_c=w_k\bar r_L$',ha='left',va='center',fontsize=18,color=st.INK)
    right.text(.02,.38,r'$w_k=\frac{k\hat\tau^2}{k\hat\tau^2+\hat\sigma^2}$',ha='left',va='center',fontsize=18,color=st.INK)
    right.text(.02,.14,r'$\hat y=f(x)+\hat b_c$',ha='left',va='center',fontsize=18,color=st.INK)

def draw_stacking_overview(fig,ss):
    before=set(fig.axes);f2.draw_f(fig,ss)
    for ax in [a for a in fig.axes if a not in before]:
        ax.set_xticklabels([x.get_text() for x in ax.get_xticklabels()],rotation=0,ha='center')
        ax.set_ylim(-.025,.62)

class Compact(st.Composite):
    def __init__(self,name,stem,height):
        super().__init__(name,stem,height,width=13.5,margins=(.035,.995,.012,.99))
    def finish(self,standalone=True):
        report=st.refine_layout(self.fig,self.records,iters=28)
        W=self.fig.get_figwidth()*72
        for rec in report:
            self.fig.text(rec['outer'][0]-8/W,rec['outer'][3],rec['panel'],ha='right',va='top',
                          fontsize=18,fontweight='normal',color=st.INK,zorder=100)
        st.regular_text(self.fig)
        report_json={'layout':report,'width_inches':self.fig.get_figwidth(),'height_inches':self.fig.get_figheight(),
                     'outer_gap_inches':.30,'prior_outer_row_gap_inches':.55,'prior_outer_column_gap_inches':.62,
                     'panel_count':len(self.records)}
        (QA/f'{self.name}_layout.json').write_text(json.dumps(report_json,indent=2),encoding='utf-8')
        st.save_pair(self.fig,DEST/self.name,self.stem)
        if standalone:
            sub=DEST/self.name/'subfigures';sub.mkdir(parents=True,exist_ok=True)
            W,H=self.fig.get_size_inches()
            for rec in self.records:
                box=rec['target']
                render_standalone(rec['draw'],(box.width*W,box.height*H),sub,f'{self.stem}_{rec["key"]}',
                                  extra_left=(self.name=='Fig7' and rec['key']=='a'))
            (sub/'README.md').write_text('Each panel is rendered independently from its draw function on a standalone canvas; no panel is cropped from the combined figure.\n',encoding='utf-8')
        plt.close(self.fig)
        return report

def render_standalone(draw,size,folder,stem,extra_left=False):
    if not extra_left:
        return st.render_standalone(draw,size,folder,stem)
    # PDF's rotated mathtext superscript extends ~2 pt farther than its Agg bbox.
    # Keep a small genuine canvas margin for the narrow variance-ratio panel.
    fig=plt.figure(figsize=size)
    gs=fig.add_gridspec(1,1,left=.03,right=.99,bottom=.01,top=.99)
    before=set(fig.axes);draw(fig,gs[0])
    axes=[a for a in fig.axes if a not in before]
    st.refine_layout(fig,[dict(key='',axes=axes,target=gs[0].get_position(fig).frozen())])
    st.regular_text(fig);st.save_pair(fig,folder,stem);plt.close(fig)

def row(cf,ss,ratios,draws,start):
    g=st.hgrid(ss,ratios,gap=.30)
    for j,draw in enumerate(draws):cf.panel(chr(97+start+j),g[j],draw)

def render(name,standalone=True):
    global CURRENT
    st.setup();draws=builders(name);CURRENT=name
    # Write source tables prepared while capturing supplementary closures.
    for origin,files in SOURCES.items():
        if any(x.startswith(origin.replace('Fig','')) for x in MAPPING[name]['old']):
            for fname,df in files.items():route_source(origin,fname,df)
    spec=MAPPING[name];cf=Compact(name,spec['stem'],spec['height'])
    if name=='Fig1':
        rows=cf.grid([1.15,1.18,.8],gap=.30)
        cf.panel('a',rows[0],draws[0]);row(cf,rows[1],[1.5,.8,1.2],draws[1:4],1);row(cf,rows[2],[1,1.1],draws[4:],4)
    elif name=='Fig2':
        rows=cf.grid([1.3,.9,1.05],gap=.30)
        cf.panel('a',rows[0],draws[0]);row(cf,rows[1],[1.8,1.2,.9],draws[1:4],1);row(cf,rows[2],[1.3,1],draws[4:],4)
    elif name=='Fig3':
        rows=cf.grid([.8,1,1],gap=.30)
        cf.panel('a',rows[0],draws[0]);row(cf,rows[1],[1,1],draws[1:3],1);row(cf,rows[2],[1,1],draws[3:5],3)
    elif name=='Fig4':
        rows=cf.grid([1,1,1.05],gap=.30)
        row(cf,rows[0],[1,1.4],draws[:2],0);row(cf,rows[1],[1,1,1.2],draws[2:5],2);row(cf,rows[2],[1.1,1.2],draws[5:],5)
    elif name=='Fig5':
        rows=cf.grid([.85,1.1],gap=.30)
        row(cf,rows[0],[.85,1.15],draws[:2],0);cf.panel('c',rows[1],draws[2])
    elif name=='Fig6':
        rows=cf.grid([1.12,.85,1],gap=.30)
        cf.panel('a',rows[0],draws[0]);cf.panel('b',rows[1],draws[1]);row(cf,rows[2],[.8,1,1.8],draws[2:],2)
    elif name=='Fig7':
        rows=cf.grid([1,.85,.9],gap=.30)
        row(cf,rows[0],[.75,1,1,1],draws[:4],0);cf.panel('e',rows[1],draws[4]);cf.panel('f',rows[2],draws[5])
    elif name=='Fig8':
        rows=cf.grid([1.05,.9,1.05,.9,1.05],gap=.30)
        for i in range(4):cf.panel(chr(97+i),rows[i],draws[i])
        row(cf,rows[4],[1,1],draws[4:],4)
    elif name=='Fig9':
        rows=cf.grid([1,1],gap=.30)
        row(cf,rows[0],[1.15,.9,1],draws[:3],0);row(cf,rows[1],[1,1,1],draws[3:],3)
    elif name=='Fig10':
        rows=cf.grid([1.15,.92,1,1.0],gap=.30)
        row(cf,rows[0],[1,1.08],draws[:2],0);row(cf,rows[1],[1.45,1],draws[2:4],2)
        row(cf,rows[2],[1.1,1],draws[4:6],4);row(cf,rows[3],[1.25,.9],draws[6:],6)
    elif name=='FigS1':
        rows=cf.grid([1.06,1],gap=.30);row(cf,rows[0],[1,1,1],draws[:3],0);row(cf,rows[1],[1,1],draws[3:],3)
    report=cf.finish(standalone)
    # Every notebook runs the canonical composition entry point, including standalone panels.
    nb={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':[f'%run ../../../scripts/figures/generate_ten_main_figures.py --only {name}\n']}],
         'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    (DEST/name/(spec['stem']+'_jupyter_code.ipynb')).write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print(name,'panels',len(draws),'max residual pt',round(max(x['max_residual_pt'] for x in report),3),flush=True)
    CURRENT=None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='+',default=list(MAPPING));p.add_argument('--combined-only',action='store_true');args=p.parse_args()
    for name in args.only:render(name,not args.combined_only)
