from distutils.core import setup
setup(
  name = 'nelder',
  packages = ['nelder', 'nelder.glm'], 
  version = '0.0.1',
  description = 'Next generation statistics software',
  author = 'Ross Taylor',
  author_email = 'rj-taylor@live.co.uk',
  url = 'https://github.com/rjt1990/nelder', 
  download_url = 'https://github.com/rjt1990/nelder/0.0.1', 
  keywords = ['statistics'],
  license = 'BSD',
  install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'tensorflow']
)