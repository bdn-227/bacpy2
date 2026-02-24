
# import packages
import polars as pl
from os.path import abspath, dirname
import importlib.resources as pkg_resources

#taxonomy = pl.read_csv(f"{dirname(abspath(__file__))}/taxonomy.tsv", separator="\t")
#taxonomy2 = pl.read_csv(f"{dirname(abspath(__file__))}/taxonomy2.tsv", separator="\t")


# create a taxonomic mapping file that spans taxonomy from kingdom --> strainID, while also including
# medium and blank measurements
# as well as each row with reduced taxonomic depth
# i.e.:
    # kingdom --> strainID
    # kingdom --> genus
    # kingdom --> family
    # kingdom --> order
    # kingdom --> class
    # kingdom --> phylum
    # kingdom


taxonomic_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "strainID"]

with pkg_resources.path("bacpy", "taxonomy2.tsv") as p:
    taxo_table = (pl.read_csv(str(p), separator="\t")
                        .select(taxonomic_levels)
                        .fill_null("unknown")
                        .with_columns((level + "_" + pl.col(level)).alias(level) for level in taxonomic_levels)
                        )

taxonomy_dict = {level: taxo_table.select(taxonomic_levels[0:index+1]).unique() for index, level in enumerate(taxonomic_levels)}



# taxonomy table
with pkg_resources.path("bacpy", "taxonomy.tsv") as p:
    taxonomy_df = pl.read_csv(str(p), separator="\t")
    taxonomy_df = taxonomy_df.select([tax for tax in taxonomic_levels if tax in taxonomy_df.columns]).fill_null("n.d.")

with pkg_resources.path("bacpy", "taxonomy2.tsv") as p:
    taxonomy_df2 = pl.read_csv(str(p), separator="\t")
    taxonomy_df2 = taxonomy_df2.select([tax for tax in taxonomic_levels if tax in taxonomy_df2.columns]).fill_null("n.d.")