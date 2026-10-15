"""Optional notebook controls; the ordinary code cells work without these panels."""
import copy
import ipywidgets as w
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import pandas as pd
from hydraulics import (solve4,network_plot4,plot_profiles4,profile4,export_result,DownloadLink,
    lab5_config,initial5,size5,hardy_cross5,network_plot5,plot_history5)

class Lab4Workbench:
    """Properties -> Compute -> inspect -> Save, with stale-result protection."""
    def __init__(self,cfg,log):
        self.cfg=copy.deepcopy(cfg);self.log=log;self.last_result=None;self.loading=False
        self.pipe=w.Dropdown(options=list(cfg['pipes'].index),description='Pipe')
        self.length=w.BoundedFloatText(min=.001,max=1e6,description='Length (m)')
        self.diameter=w.BoundedFloatText(min=1,max=10000,description='D (mm)')
        self.factor=w.BoundedFloatText(min=.0001,max=.2,description='Darcy f')
        self.heads={r:w.FloatText(value=h,description=f'{r} H (m)') for r,h in cfg['reservoir_heads'].items()}
        self.bends={r:w.BoundedFloatText(value=k,min=0,max=100,description=f'{r} bend K') for r,k in cfg['bend_K'].items()}
        self.minors=w.Checkbox(value=cfg['include_minor'],description='Include local losses')
        self.tee=w.BoundedFloatText(value=cfg['tee_K'],min=0,max=100,description='P-2 tee K')
        self.status=w.HTML('<b>Ready.</b> Edit properties, then Compute. Geometry elevations stay at the input baseline.')
        self.run_button=w.Button(description='Compute',button_style='primary',icon='play')
        self.save_button=w.Button(description='Save results ZIP',icon='download',disabled=True)
        self.outputs=[w.Output() for _ in range(5)]
        self.tabs=w.Tab(children=self.outputs)
        for i,title in enumerate(['Model view','Pipe / loss tables','HGL and EGL','Pressure screen','Run log']):self.tabs.set_title(i,title)
        self.pipe.observe(self.load_pipe,names='value');self.load_pipe()
        for control in [self.length,self.diameter,self.factor,*self.heads.values(),*self.bends.values(),self.minors,self.tee]:control.observe(self.dirty,names='value')
        self.run_button.on_click(self.compute);self.save_button.on_click(self.save)
        self.widget=w.VBox([w.HTML('<h3>Lab 4 model workbench</h3><p>Fixed Darcy factors; bend K values are editable teaching assumptions. Pipe names select existing elements.</p>'),
            w.HBox([self.pipe,self.length]),w.HBox([self.diameter,self.factor]),
            w.HBox(list(self.heads.values()),layout=w.Layout(flex_flow='row wrap')),w.HBox(list(self.bends.values()),layout=w.Layout(flex_flow='row wrap')),
            w.HBox([self.minors,self.tee]),w.HBox([self.run_button,self.save_button]),self.status,self.tabs])
    def load_pipe(self,change=None):
        self.loading=True;p=self.cfg['pipes'].loc[self.pipe.value]
        self.length.value=p.L_m;self.diameter.value=p.D_m*1000;self.factor.value=p.f_D;self.loading=False
    def dirty(self,change=None):
        if self.loading:return
        self.cfg['pipes'].loc[self.pipe.value,['L_m','D_m','f_D']]=[self.length.value,self.diameter.value/1000,self.factor.value]
        self.last_result=None;self.save_button.disabled=True
        self.status.value='<b>Inputs changed.</b> Earlier displays are stale; click Compute before using or saving them.'
    def compute(self,_=None):
        for out in self.outputs:out.clear_output(wait=False)
        self.last_result=None;self.save_button.disabled=True;self.run_button.disabled=True
        try:
            cfg=copy.deepcopy(self.cfg);cfg['reservoir_heads']={r:x.value for r,x in self.heads.items()}
            cfg['bend_K']={r:x.value for r,x in self.bends.items()};cfg['tee_K']=self.tee.value;cfg['include_minor']=self.minors.value
            cfg['friction_mode']='fixed'
            # Water-level scenarios do not move the buried pipe centerline.
            if 'geometry_elevations' not in cfg:
                from hydraulics import elevations4
                cfg['geometry_elevations']=elevations4(self.cfg)
            with self.outputs[4]:
                self.log.info('Workbench Compute: fixed-f scenario; saved results will include all inputs.')
                result=solve4(cfg,self.log)
            self.last_result=result
            with self.outputs[0]:fig=network_plot4(result);display(fig);plt.close(fig)
            with self.outputs[1]:display(result['pipes'].round(6));display(result['losses'].fillna(''))
            with self.outputs[2]:fig=plot_profiles4(result);display(fig);plt.close(fig)
            with self.outputs[3]:display(profile4(result).groupby('branch')[['p_abs_kPa','vapor_margin_kPa']].min().round(3))
            self.status.value=f"<b>Converged.</b> J-2 total head = {result['head_J2']:.5f} m. Mass residual = {result['mass_residual']:.2e} m³/s. Inspect all tabs."
            self.save_button.disabled=False
        except Exception as exc:
            with self.outputs[4]:self.log.exception('Compute failed. No results were saved.')
            self.status.value=f'<b>Input or calculation error:</b> {type(exc).__name__}. See Run log; correct inputs and compute again.'
        finally:self.run_button.disabled=False
    def save(self,_=None):
        if self.last_result is None:return
        with self.outputs[4]:
            figures={'network':network_plot4(self.last_result),'profiles':plot_profiles4(self.last_result)}
            path=export_result(self.last_result,'lab4_workbench',figures)
            for fig in figures.values():plt.close(fig)
            self.log.info('Saved current workbench inputs and results to %s. Download before leaving Binder.',path)
            display(DownloadLink(str(path)))

class Lab5Workbench:
    def __init__(self,cfg,log):
        self.cfg=copy.deepcopy(cfg);self.log=log;self.last_result=None
        self.mode=w.Dropdown(options=[('Manual demand table','table'),('Population daily average','population')],value=cfg['demand_mode'],description='Demand')
        self.scale=w.FloatSlider(value=1,min=.5,max=2,step=.05,description='Demand ×',continuous_update=False)
        self.rough=w.FloatSlider(value=.15,min=.01,max=1,step=.01,description='ε (mm)',continuous_update=False)
        self.diameters={pid:w.BoundedFloatText(value=p.D_m*1000,min=25,max=2000,description=pid+' mm',layout=w.Layout(width='210px')) for pid,p in cfg['pipes'].iterrows()}
        self.button=w.Button(description='Run Hardy–Cross',button_style='primary');self.save_button=w.Button(description='Save results ZIP',disabled=True)
        self.status=w.HTML('Ready. The diameter boxes start with the preliminary design; they are not automatically resized.')
        self.out=w.Output();self.button.on_click(self.compute);self.save_button.on_click(self.save)
        for control in [self.mode,self.scale,self.rough,*self.diameters.values()]:control.observe(self.dirty,names='value')
        self.widget=w.VBox([w.HTML('<h3>Lab 5 design bench</h3>'),w.HBox([self.mode,self.scale,self.rough],layout=w.Layout(flex_flow='row wrap')),
            w.GridBox(list(self.diameters.values()),layout=w.Layout(grid_template_columns='repeat(auto-fit, minmax(210px, 1fr))')),
            w.HBox([self.button,self.save_button]),self.status,self.out])
    def dirty(self,_):
        self.last_result=None;self.save_button.disabled=True;self.status.value='Inputs changed. Run Hardy–Cross again; previous displays are stale.'
    def compute(self,_=None):
        self.button.disabled=True;self.save_button.disabled=True;self.last_result=None
        with self.out:
            clear_output(wait=True)
            try:
                cfg=copy.deepcopy(self.cfg);d=lab5_config(self.mode.value)['demand'];cfg['demand']={n:v*self.scale.value for n,v in d.items()};cfg['demand_mode']=self.mode.value
                cfg['pipes']['eps_m']=self.rough.value/1000
                cfg['pipes']['D_m']=[x.value/1000 for x in self.diameters.values()]
                self.last_result=hardy_cross5(cfg,log=self.log)
                display(self.last_result['pipes'].round(4));display(self.last_result['nodes'].round(4))
                for fig in [network_plot5(self.last_result),plot_history5(self.last_result)]:display(fig);plt.close(fig)
                self.save_button.disabled=False;self.status.value='Converged. Compare velocities, pressures, reservoir exchange, and residuals before accepting this design.'
            except Exception:
                self.log.exception('Design run failed; no result saved.');self.status.value='Calculation failed. Inspect the log and revise inputs.'
            finally:self.button.disabled=False
    def save(self,_=None):
        if self.last_result is None:return
        with self.out:
            from hydraulics import plot_profiles5
            figs={'network':network_plot5(self.last_result),'profiles':plot_profiles5(self.last_result)}
            path=export_result(self.last_result,'lab5_design',figs)
            for fig in figs.values():plt.close(fig)
            self.log.info('Saved %s. Download this archive before ending Binder.',path);display(DownloadLink(str(path)))


class PipeExplorer5:
    """Inspect the baseline (or any explicit result) without recomputing it."""
    def __init__(self,result):
        self.result=result
        self.pipe=w.Dropdown(options=list(result['pipes'].index),value='AB',description='Inspect pipe:')
        self.output=w.Output()
        self.pipe.observe(self.draw,names='value')
        self.widget=w.VBox([w.HTML('<b>Pipe detail explorer</b> — this view uses the supplied solved case.'),self.pipe,self.output])
        self.draw()
    def draw(self,_=None):
        from hydraulics import plot_pipe_detail5
        with self.output:
            clear_output(wait=True)
            fig=plot_pipe_detail5(self.result,self.pipe.value);display(fig);plt.close(fig)
