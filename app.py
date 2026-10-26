from io import BytesIO
from datetime import datetime
import json
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

st.set_page_config(page_title='Molecule Atlas', page_icon='🧬', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#f4f2ec;color:#17231f}.block-container{max-width:1320px;padding-top:1.5rem}
.atlas{background:#182d27;color:#f7f4eb;padding:1.6rem 1.8rem;border-radius:7px 26px 7px 26px}.kicker{font-family:'DM Mono',monospace;color:#c8e56a;font-size:.72rem;letter-spacing:.17em}.atlas h1{font-family:'Space Grotesk',sans-serif;font-size:2.6rem;margin:.3rem 0;color:#f7f4eb}.atlas p{color:#d1d8d0;margin:0}.spec{background:#fffdf7;border:1px solid #deded3;border-radius:15px;padding:1rem;min-height:90px}.lab{color:#65736a;text-transform:uppercase;letter-spacing:.12em;font-family:'DM Mono',monospace;font-size:.68rem}.val{color:#182d27;font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;margin-top:.35rem}div[data-testid="stMetric"]{background:#fffdf7;border:1px solid #deded3;padding:12px;border-radius:13px}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="atlas"><div class="kicker">FIELD NOTES / ORGANIC CHEMISTRY / VOL. 01</div><h1>MOLECULE ATLAS<span style="color:#c8e56a">.</span></h1><p>A visual chemistry notebook for exploring structure, molecular fingerprints, functional groups and compound context — with a student-ready dossier export.</p></div>''', unsafe_allow_html=True)
st.write('')

DB={
'Glucose':dict(smiles='OCC1OC(O)C(O)C(O)C1O',kind='Monosaccharide · aldohexose',mp='≈146 °C (decomposes; form-dependent)',bp='No normal boiling point; decomposes',bond='Predominantly covalent single bonds in the selected cyclic representation.',angle='Saturated carbon centres are approximately tetrahedral (~109.5°); ring geometry changes local angles.',story='A compact energy currency in living systems. In water, glucose exists as a changing mixture of cyclic anomers with a small open-chain fraction.',isomers='α- and β-D-glucose are anomers. D- and L-glucose are enantiomers. Fructose shares the formula but differs in connectivity; galactose is a stereoisomer.',groups=['Hydroxyl groups','Cyclic hemiacetal']),
'Sucrose':dict(smiles='OCC1OC(OC2C(O)C(O)C(OC2CO)O)C(O)C(O)C1O',kind='Disaccharide · non-reducing sugar',mp='≈186 °C (decomposition/caramelization; varies)',bp='No normal boiling point; decomposes',bond='Covalent bonds, including an oxygen-containing glycosidic bridge between sugar units.',angle='Most saturated carbon centres are approximately tetrahedral; the glycosidic linkage is conformationally flexible.',story='Sucrose joins glucose and fructose through a glycosidic linkage. Because both anomeric centres participate, it is classed as a non-reducing sugar.',isomers='Maltose and lactose have the same molecular formula but different monosaccharide linkages/connectivity.',groups=['Hydroxyl groups','Glycosidic acetal/ether linkage']),
'Adenine':dict(smiles='C1=NC2=C(N1)C(=NC=N2)N',kind='Purine nucleobase · 6-aminopurine',mp='≈360–365 °C (decomposition/sublimation reported)',bp='No reliable normal boiling point; decomposes',bond='Conjugated aromatic ring bonds with delocalized π electrons; exocyclic amino group.',angle='Ring atoms are broadly trigonal planar (~120°), with local differences by atom and tautomer.',story='A nucleobase used in DNA and RNA. It pairs with thymine in DNA and uracil in RNA in canonical Watson–Crick pairing.',isomers='Amino–imino tautomerism is possible. Other aminopurine positional isomers can share the formula while changing substitution position.',groups=['Fused nitrogen-containing aromatic rings','Primary amino group']),
'Guanine':dict(smiles='C1=NC2=C(N1)C(=O)NC(=N2)N',kind='Purine nucleobase · 2-amino-6-oxopurine',mp='High; around 360 °C with decomposition reported',bp='No reliable normal boiling point; decomposes',bond='Conjugated heteroaromatic ring system, carbonyl C=O and amino functionality.',angle='Ring atoms are broadly trigonal planar (~120°); values vary with tautomer and local bonding.',story='Guanine is a DNA/RNA nucleobase and forms canonical Watson–Crick pairs with cytosine through three hydrogen bonds.',isomers='Keto–enol and amino–imino tautomerism; positional isomers can alter the placement of amino/oxo groups.',groups=['Fused nitrogen-containing aromatic rings','Carbonyl group','Primary amino group']),
'Aniline':dict(smiles='Nc1ccccc1',kind='Aromatic amine · aminobenzene',mp='≈−6 °C',bp='≈184 °C at 1 atm',bond='Aromatic delocalized π system in the phenyl ring; C–N and N–H covalent bonds.',angle='Phenyl ring carbons are approximately trigonal planar (~120°); nitrogen geometry is influenced by conjugation.',story='Aniline is an industrial intermediate for dyes, polymers and other chemicals. It is toxic and requires appropriate laboratory controls.',isomers='Formula-identical isomers must retain C₆H₇N but change connectivity; compare database structures before naming specific isomers.',groups=['Benzene ring','Primary aromatic amine'])}

with st.sidebar:
 st.markdown('## Notebook setup')
 student=st.text_input('Student name',placeholder='Full name'); reg=st.text_input('Registration number',placeholder='Registration number')
 course=st.text_input('Course / class',value='Chemistry'); institution=st.text_input('Institution',value='SRM Institute of Science and Technology')
 st.divider(); mode=st.radio('Workspace',['Compound library','Custom SMILES','Compare compounds'])
 chosen=st.selectbox('Featured compound',list(DB))
 if mode=='Custom SMILES':
  name=st.text_input('Structure label',value='My molecule'); smiles=st.text_area('Paste SMILES',placeholder='e.g. Nc1ccccc1')
  meta=dict(kind='User-supplied structure',mp='Not supplied',bp='Not supplied',bond='See bond inventory.',angle='Qualitative only; exact angles require a 3D conformer or experimental structure.',story='Custom structure supplied by the user.',isomers='No isomer enumeration is performed for custom inputs.',groups=['Functional groups require interpretation.'])
 else:
  name=chosen; smiles=DB[chosen]['smiles']; meta=DB[chosen]
 compare=st.multiselect('Compare compounds',list(DB),default=['Glucose','Sucrose']) if mode=='Compare compounds' else []

def analyse(s):
 m=Chem.MolFromSmiles(s)
 if m is None:return None
 bc={'Single':0,'Double':0,'Triple':0,'Aromatic':0,'Other':0}
 for b in m.GetBonds():
  t=b.GetBondType(); k='Single' if t==Chem.BondType.SINGLE else 'Double' if t==Chem.BondType.DOUBLE else 'Triple' if t==Chem.BondType.TRIPLE else 'Aromatic' if t==Chem.BondType.AROMATIC else 'Other'; bc[k]+=1
 im=Draw.MolToImage(m,size=(760,500)); bio=BytesIO(); im.save(bio,format='PNG')
 return dict(m=m,canonical=Chem.MolToSmiles(m,canonical=True),formula=rdMolDescriptors.CalcMolFormula(m),mw=Descriptors.MolWt(m),exact=Descriptors.ExactMolWt(m),logp=Descriptors.MolLogP(m),tpsa=CalcTPSA(m),hbd=rdMolDescriptors.CalcNumHBD(m),hba=rdMolDescriptors.CalcNumHBA(m),rot=rdMolDescriptors.CalcNumRotatableBonds(m),rings=rdMolDescriptors.CalcNumRings(m),charge=Chem.GetFormalCharge(m),chiral=len(Chem.FindMolChiralCenters(m,includeUnassigned=True)),heavy=m.GetNumHeavyAtoms(),bonds=bc,png=bio.getvalue())

if mode=='Custom SMILES' and not smiles.strip(): st.info('Paste a SMILES string in the sidebar to start.'); st.stop()
r=analyse(smiles)
if r is None: st.error('SMILES could not be parsed. Check the string and try again.'); st.stop()
if mode=='Compare compounds':
 st.markdown('### 01 / Comparison bench')
 if len(compare)<2: st.info('Choose at least two compounds in the sidebar to compare their molecular fingerprints.')
 else:
  rows=[]
  for n in compare:
   x=analyse(DB[n]['smiles']); rows.append({'Compound':n,'Formula':x['formula'],'Molar mass (g/mol)':round(x['mw'],3),'Heavy atoms':x['heavy'],'H-bond donors':x['hbd'],'H-bond acceptors':x['hba'],'TPSA (Å²)':round(x['tpsa'],1),'Rings':x['rings']})
  st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
 st.divider()
st.markdown(f'### 02 / Specimen card · {name}')
cols=st.columns(4)
for c,l,v in zip(cols,['MOLECULAR FORMULA','MOLAR MASS','HEAVY ATOMS','RING COUNT'],[r['formula'],f"{r['mw']:.3f} g/mol",str(r['heavy']),str(r['rings'])]): c.markdown(f'<div class="spec"><div class="lab">{l}</div><div class="val">{v}</div></div>',unsafe_allow_html=True)
st.write('')
t1,t2,t3,t4=st.tabs(['Structure board','Property fingerprint','Chemistry notes','Export dossier'])
with t1:
 a,b=st.columns([1.1,.9]); a.image(r['png'],use_container_width=True,caption='2D depiction generated from parsed SMILES')
 b.markdown('#### Input SMILES'); b.code(smiles); b.markdown('#### Canonical SMILES'); b.code(r['canonical']); b.markdown('#### Bond census'); b.dataframe(pd.DataFrame([{'Bond type':k,'Count':v} for k,v in r['bonds'].items() if v]),hide_index=True,use_container_width=True); b.markdown('#### Geometry note'); b.write(meta['angle'])
with t2:
 st.markdown('#### Calculated fingerprint'); cols=st.columns(4)
 for c,l,v in zip(cols,['Exact mass','LogP estimate','Polar surface','Formal charge'],[f"{r['exact']:.5f} Da",f"{r['logp']:.2f}",f"{r['tpsa']:.1f} Å²",str(r['charge'])]): c.metric(l,v)
 cols=st.columns(4)
 for c,l,v in zip(cols,['H-bond donors','H-bond acceptors','Rotatable bonds','Chiral centres'],[r['hbd'],r['hba'],r['rot'],r['chiral']]): c.metric(l,v)
 a,b=st.columns(2); a.markdown(f'<div class="spec"><div class="lab">MELTING POINT</div><div class="val">{meta["mp"]}</div></div>',unsafe_allow_html=True); b.markdown(f'<div class="spec"><div class="lab">BOILING POINT</div><div class="val">{meta["bp"]}</div></div>',unsafe_allow_html=True)
 st.warning('Melting/boiling values are reference notes, not measurements. Some compounds decompose before a normal boiling point can be measured.')
with t3:
 st.markdown('#### The chemistry in one minute'); st.write(meta['story']); st.markdown('#### Functional-group watchlist')
 for g in meta['groups']: st.markdown(f'- `{g}`')
 st.markdown('#### Isomer & stereochemistry notebook'); st.write(meta['isomers']); st.markdown('#### Bonding interpretation'); st.write(meta['bond'])
with t4:
 st.markdown('### Build a field dossier'); st.write('PDF combines student details, structure, molecular fingerprint, bond census, property notes and isomer context.')
 def make_pdf():
  buff=BytesIO(); doc=SimpleDocTemplate(buff,pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=15*mm,bottomMargin=15*mm); ss=getSampleStyleSheet()
  ss.add(ParagraphStyle(name='AtlasTitle',parent=ss['Title'],fontSize=23,textColor=colors.HexColor('#18352c'),spaceAfter=5)); ss.add(ParagraphStyle(name='AtlasSection',parent=ss['Heading2'],fontSize=12,textColor=colors.HexColor('#526b2c'),spaceBefore=11,spaceAfter=5)); ss.add(ParagraphStyle(name='TinyAtlas',parent=ss['BodyText'],fontSize=8,leading=10,wordWrap='CJK'))
  story=[Paragraph('MOLECULE ATLAS',ss['AtlasTitle']),Paragraph('Chemical specimen dossier · generated '+datetime.now().strftime('%d %B %Y, %H:%M'),ss['TinyAtlas']),Spacer(1,8)]
  idata=[['Student',student or 'Not entered','Registration',reg or 'Not entered'],['Course',course or 'Not entered','Institution',institution or 'Not entered']]; tb=Table(idata,colWidths=[23*mm,56*mm,25*mm,67*mm]); tb.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#e8eddc')),('BACKGROUND',(2,0),(2,-1),colors.HexColor('#e8eddc')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cbd2c3')),('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story += [tb,Paragraph('01 / Specimen identity',ss['AtlasSection']),Paragraph(f"<b>Name:</b> {name}<br/><b>Class:</b> {meta['kind']}<br/><b>Formula:</b> {r['formula']}<br/><b>Molar mass:</b> {r['mw']:.4f} g/mol<br/><b>Exact mass:</b> {r['exact']:.5f} Da",ss['BodyText']),Paragraph('02 / Structure board',ss['AtlasSection']),Image(BytesIO(r['png']),width=145*mm,height=95*mm),Paragraph('<b>Input SMILES:</b> '+smiles.replace('&','&amp;').replace('<','&lt;'),ss['TinyAtlas']),Paragraph('<b>Canonical SMILES:</b> '+r['canonical'].replace('&','&amp;').replace('<','&lt;'),ss['TinyAtlas']),Paragraph('03 / Molecular fingerprint',ss['AtlasSection'])]
  fp=[['Descriptor','Value','Descriptor','Value'],['Exact mass',f"{r['exact']:.5f} Da",'LogP estimate',f"{r['logp']:.2f}"],['TPSA',f"{r['tpsa']:.2f} Å²",'Formal charge',str(r['charge'])],['H-bond donors',str(r['hbd']),'H-bond acceptors',str(r['hba'])],['Rotatable bonds',str(r['rot']),'Ring count',str(r['rings'])],['Chiral centres',str(r['chiral']),'Heavy atoms',str(r['heavy'])]]; ft=Table(fp,colWidths=[34*mm,43*mm,34*mm,60*mm]); ft.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18352c')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#cbd2c3')),('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story += [ft,Paragraph('04 / Bond census & physical properties',ss['AtlasSection']),Paragraph('<br/>'.join(f'<b>{k} bonds:</b> {v}' for k,v in r['bonds'].items())+f"<br/><b>Bonding:</b> {meta['bond']}<br/><b>Geometry:</b> {meta['angle']}<br/><b>Melting point:</b> {meta['mp']}<br/><b>Boiling point:</b> {meta['bp']}",ss['BodyText']),Paragraph('05 / Context & isomers',ss['AtlasSection']),Paragraph(meta['story']+'<br/><br/><b>Isomer notes:</b> '+meta['isomers'],ss['BodyText']),Spacer(1,9),Paragraph('Data note: Structure-derived descriptors were calculated from the entered SMILES using RDKit. Physical properties and isomer notes are educational references and should be independently verified before formal submission. Exact bond angles require a defined 3D geometry or experimental structure.',ss['TinyAtlas'])]; doc.build(story); buff.seek(0); return buff.getvalue()
 pdf=make_pdf(); slug=''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_') or 'molecule'
 st.download_button('Download specimen dossier (PDF)',pdf,file_name=f'{slug}_atlas_dossier.pdf',mime='application/pdf',use_container_width=True)
 payload={'student':student,'registration_number':reg,'course':course,'institution':institution,'compound':name,'input_smiles':smiles,'canonical_smiles':r['canonical'],'formula':r['formula'],'molar_mass_g_mol':r['mw'],'exact_mass_da':r['exact'],'bond_counts':r['bonds'],'logp_estimate':r['logp'],'TPSA_A2':r['tpsa'],'HBD':r['hbd'],'HBA':r['hba'],'ring_count':r['rings'],'melting_point_reference':meta['mp'],'boiling_point_reference':meta['bp']}
 st.download_button('Download raw data (JSON)',json.dumps(payload,indent=2),file_name=f'{slug}_atlas_data.json',mime='application/json',use_container_width=True)
st.divider(); st.caption('MOLECULE ATLAS · Educational chemistry tool. Verify stereochemistry and reference properties before submitting academic work.')
