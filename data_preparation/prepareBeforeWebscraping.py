import pandas as pd

tf_data = pd.read_csv("./bangkok_traffy.csv")
tf_df = pd.DataFrame(tf_data)

flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]

tf_org_agg = tf_df['organization'].unique()
tf_org = pd.Series(flat_map(lambda r: f'{r}'.split(","), tf_org_agg))
tf_org = tf_org.apply(lambda r: r.strip())
tf_org = pd.DataFrame(tf_org.unique())

tf_org.to_csv('unique_org.csv', index=False)