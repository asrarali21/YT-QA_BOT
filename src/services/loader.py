from langchain_community.document_loaders import YoutubeLoader




def load_youtube_video(url:str):
      loader = YoutubeLoader.from_youtube_url(url)

      docs_loaded = loader.load()

      return docs_loaded