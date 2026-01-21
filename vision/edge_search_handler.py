"""
Edge Search Handler - Dynamic Web Search Support
Supports 20+ websites with intelligent fallback for unknown domains
Uses urllib.parse for proper URL encoding with quote_plus
"""

import subprocess
import urllib.parse
import os
import platform
import logging

logger = logging.getLogger(__name__)


class EdgeSearchHandler:
    """
    Handles web searches using Microsoft Edge browser
    Supports dynamic website search with 20+ pre-configured patterns
    Falls back to Google for unknown websites
    """

    def __init__(self):
        """Initialize EdgeSearchHandler with website patterns"""
        self.edge_path = self._find_edge_path()
        
        # Pre-configured website patterns (20+)
        # Each pattern contains {query} placeholder for search term
        self.website_patterns = {
    'google': 'https://www.google.com/search?q={query}',
    'youtube': 'https://www.youtube.com/results?search_query={query}',
    'wikipedia': 'https://en.wikipedia.org/wiki/Special:Search?search={query}',
    'github': 'https://github.com/search?q={query}',
    'stackoverflow': 'https://stackoverflow.com/search?q={query}',
    'reddit': 'https://www.reddit.com/search/?q={query}',
    'amazon': 'https://www.amazon.com/s?k={query}',
    'twitter': 'https://twitter.com/search?q={query}',
    'linkedin': 'https://www.linkedin.com/search/results/all/?keywords={query}',
    'instagram': 'https://www.instagram.com/explore/tags/{query}/',
    'pinterest': 'https://www.pinterest.com/search/?q={query}',
    'medium': 'https://medium.com/search?q={query}',
    'quora': 'https://www.quora.com/search?q={query}',
    'ebay': 'https://www.ebay.com/sch/i.html?_nkw={query}',
    'imdb': 'https://www.imdb.com/find?q={query}',
    'bing': 'https://www.bing.com/search?q={query}',
    'duckduckgo': 'https://duckduckgo.com/?q={query}',
    'facebook': 'https://www.facebook.com/search/top/?q={query}',
    'gmail': 'https://mail.google.com/mail/u/0/#search/{query}',
    'netflix': 'https://www.netflix.com/search?q={query}',
    'spotify': 'https://open.spotify.com/search/{query}',
    'chatgpt': 'https://chat.openai.com/?q={query}',
    'whatsapp': 'https://web.whatsapp.com/',
    'x': 'https://x.com/search?q={query}',
    'hotstar': 'https://www.hotstar.com/in/search?q={query}',
    'flipkart': 'https://www.flipkart.com/search?q={query}',
    'cricbuzz': 'https://www.cricbuzz.com/search?q={query}',
    'paytm': 'https://paytm.com/search?q={query}',
    'timesofindia': 'https://timesofindia.indiatimes.com/search.cms?query={query}',
    'googleco': 'https://www.google.co.in/search?q={query}',
    'justdial': 'https://www.justdial.com/search?q={query}',
    'jiosaavn': 'https://www.jiosaavn.com/search/{query}',
    'hindustantimes': 'https://www.hindustantimes.com/search?query={query}',
    'indianexpress': 'https://indianexpress.com/?s={query}',
    'espncricinfo': 'https://www.espncricinfo.com/search/?q={query}',
    'bookmyshow': 'https://in.bookmyshow.com/explore/search?q={query}',
    'zeenews': 'https://zeenews.india.com/search?query={query}',
    'snapchat': 'https://www.snapchat.com/add?q={query}',
    'telegram': 'https://t.me/s/{query}',
    'magicbricks': 'https://www.magicbricks.com/search?q={query}',
    'naukri': 'https://www.naukri.com/{query}-jobs',
    'shaadi': 'https://www.shaadi.com/search?query={query}',
    'policybazaar': 'https://www.policybazaar.com/search?q={query}',
    'myntra': 'https://www.myntra.com/search?q={query}',
    'zomato': 'https://www.zomato.com/search?query={query}',
    'swiggy': 'https://www.swiggy.com/search?q={query}',
    'aajtak': 'https://www.aajtak.in/search?q={query}',
    'ndtv': 'https://www.ndtv.com/search?q={query}',
    'moneycontrol': 'https://www.moneycontrol.com/search/?query={query}',
    'gaana': 'https://gaana.com/search/{query}',
    'yatra': 'https://www.yatra.com/search?q={query}',
    '99acres': 'https://www.99acres.com/search?q={query}',
    'cleartrip': 'https://www.cleartrip.com/search?q={query}',
    'makemytrip': 'https://www.makemytrip.com/search?q={query}',
    'olx': 'https://www.olx.in/search?q={query}',
    'carwale': 'https://www.carwale.com/search/?q={query}',
    'bikedekho': 'https://www.bikedekho.com/search?q={query}',
    'housing': 'https://housing.com/search?q={query}',
    'bigbasket': 'https://www.bigbasket.com/search/?q={query}',
    'grofers': 'https://www.blinkit.com/search?q={query}',
    'reliancedigital': 'https://www.reliancedigital.in/search?q={query}',
    'vijaysales': 'https://www.vijaysales.com/search?q={query}',
    'croma': 'https://www.croma.com/c/search?q={query}',
    'indiamart': 'https://dir.indiamart.com/search.mp?ss={query}',
    'tradeindia': 'https://www.tradeindia.com/search?q={query}',
    'sulekha': 'https://www.sulekha.com/search?q={query}',
    'mouthshut': 'https://www.mouthshut.com/search?q={query}',
    'whatmobile': 'https://whatmobile.net.in/search?q={query}',
    'hindisahityadarpan': 'https://hindisahityadarpan.in/search?q={query}',
    'abhiyojana': 'https://abhiyojana.co.in/search?q={query}',
    'pib': 'https://pib.gov.in/search.aspx?query={query}',
    'incometaxindia': 'https://incometaxindia.gov.in/search?q={query}',
    'upi': 'https://upi.gov.in/search?q={query}',
    'bhimgov': 'https://bhimgov.in/search?q={query}',
    'umang': 'https://web.umang.gov.in/search?q={query}',
    'digilocker': 'https://digilocker.gov.in/search?q={query}',
    'mygov': 'https://www.mygov.in/search?q={query}',
    'india': 'https://www.india.gov.in/search?q={query}',
    'epfindia': 'https://www.epfindia.gov.in/search?q={query}',
    'esi': 'https://www.esic.gov.in/search?q={query}',
    'passportindia': 'https://www.passportindia.gov.in/search?q={query}',
    'irctc': 'https://www.irctc.co.in/search?q={query}',
    'indianrail': 'https://indianrail.gov.in/search?q={query}',
    'airindia': 'https://www.airindia.in/search?q={query}',
    'raaga': 'https://www.raaga.com/search/{query}',
    'hungama': 'https://www.hungama.com/search/{query}',
    'wynk': 'https://wynk.in/search?q={query}',
    'jio': 'https://www.jio.com/search?q={query}',
    'airtel': 'https://www.airtel.in/search?q={query}',
    'vi': 'https://www.myvi.in/search?q={query}',
    'bsnl': 'https://www.bsnl.co.in/search?q={query}',
    'icicibank': 'https://www.icicibank.com/search?q={query}',
    'hdfcbank': 'https://www.hdfcbank.com/search?q={query}',
    'sbibank': 'https://sbi.co.in/search?q={query}',
    'axisbank': 'https://www.axisbank.com/search?q={query}',
    'kotak': 'https://www.kotak.com/search?q={query}',
    'yesbank': 'https://www.yesbank.in/search?q={query}',
    'federalbank': 'https://www.federalbank.co.in/search?q={query}',
    'ucobank': 'https://www.ucobank.com/search?q={query}',
    'canarabank': 'https://canarabank.com/search?q={query}',
    'iifl': 'https://www.iifl.com/search?q={query}',
    'bajajfinserv': 'https://www.bajajfinserv.in/search?q={query}',
    'tatacapital': 'https://www.tatacapital.com/search?q={query}',
    'lendenclub': 'https://www.lendenclub.com/search?q={query}',
    'groww': 'https://groww.in/search?q={query}',
    'zerodha': 'https://zerodha.com/search?q={query}',
    'upstox': 'https://upstox.com/search?q={query}',
    'angelone': 'https://www.angelone.in/search?q={query}',
    '5paisa': 'https://www.5paisa.com/search?q={query}',
    'dhan': 'https://dhan.co/search?q={query}',
    'icicidirect': 'https://www.icicidirect.com/search?q={query}',
    'sharekhan': 'https://www.sharekhan.com/search?q={query}',
    'motilaloswal': 'https://www.motilaloswal.com/search?q={query}',
    'choiceindia': 'https://www.choiceindia.com/search?q={query}',
    'rkglobal': 'https://www.rkglobal.net/search?q={query}',
    'indmoney': 'https://www.indmoney.com/search?q={query}',
    'smallcase': 'https://www.smallcase.com/search?q={query}',
    'fisdom': 'https://www.fisdom.com/search?q={query}',
    'bajajbroking': 'https://www.bajajbroking.in/search?q={query}',
    'hdfcsec': 'https://www.hdfcsec.com/search?q={query}',
    'kfintech': 'https://www.kfintech.com/search?q={query}',
    'nseindia': 'https://www.nseindia.com/search?q={query}',
    'bseindia': 'https://www.bseindia.com/search?q={query}',
    'mcxindia': 'https://www.mcxindia.com/search?q={query}',
    'moneybhai': 'https://moneybhai.moneycontrol.com/search?q={query}',
    'tickertape': 'https://www.tickertape.in/search?q={query}',
    'stockedge': 'https://stockedge.com/search?q={query}',
    'screener': 'https://www.screener.in/search?q={query}',
    'trendlyne': 'https://trendlyne.com/search?q={query}',
    'tradingview': 'https://www.tradingview.com/symbols/search/?query={query}',
    'investing': 'https://www.investing.com/search/?q={query}',
    'yahoofinance': 'https://in.finance.yahoo.com/search/?q={query}',
    'economictimes': 'https://economictimes.indiatimes.com/search.cms?query={query}',
    'livemint': 'https://www.livemint.com/search?q={query}',
    'financialexpress': 'https://www.financialexpress.com/search/?q={query}',
    'businesstoday': 'https://www.businesstoday.in/search?query={query}',
    'businessstandard': 'https://www.business-standard.com/search?query={query}',
    'firstpost': 'https://www.firstpost.com/search/?query={query}',
    'scroll': 'https://scroll.in/search?q={query}',
    'theprint': 'https://theprint.in/search/?query={query}',
    'opindia': 'https://www.opindia.com/?s={query}',
    'republicworld': 'https://www.republicworld.com/search?q={query}',
    'news18': 'https://www.news18.com/search/?query={query}',
    'tv9telugu': 'https://tv9telugu.com/search?q={query}',
    'sakshi': 'https://www.sakshi.com/search?q={query}',
    'andhrajyothy': 'https://www.andhrajyothy.com/search?q={query}',
    'manatelangana': 'https://www.manatelangana.com/search?q={query}',
    'greatandhra': 'https://www.greatandhra.com/search?q={query}',
    'idlebrain': 'https://www.idlebrain.com/search/?q={query}',
    '123telugu': 'https://www.123telugu.com/search?q={query}',
    'filmibeat': 'https://www.filmibeat.com/search.html?q={query}',
    'bollywoodhungama': 'https://www.bollywoodhungama.com/search/?q={query}',
    'pinkvilla': 'https://www.pinkvilla.com/search?q={query}',
    'koimoi': 'https://www.koimoi.com/search/?q={query}',
    'indiaforums': 'https://www.indiaforums.com/search?q={query}',
    'tellychakkar': 'https://www.tellychakkar.com/search/node/{query}',
    'iwmbuzz': 'https://www.iwmbuzz.com/search?q={query}',
    'gossipaddict': 'https://gossipaddict.com/search?q={query}',
    'missmalini': 'https://www.missmalini.com/search?q={query}',
    'indiacom': 'https://www.india.com/search/?q={query}',
    'jagran': 'https://www.jagran.com/search?q={query}',
    'dainikbhaskar': 'https://www.bhaskar.com/search?q={query}',
    'amarujala': 'https://www.amarujala.com/search?q={query}',
    'navbharattimes': 'https://navbharattimes.indiatimes.com/search.cms?query={query}',
    'abhiyaan': 'https://abhiyaan.com/search?q={query}',
    'livehindustan': 'https://www.livehindustan.com/search?q={query}',
    'jansatta': 'https://www.jansatta.com/?s={query}',
    'rajasthanpatrika': 'https://www.patrika.com/search?q={query}',
    'bhaskar': 'https://www.bhaskar.com/search?q={query}',
    'divyabhaskar': 'https://www.divyabhaskar.co.in/search?q={query}',
    'inshorts': 'https://www.inshorts.com/search?q={query}',
    'dailyhunt': 'https://www.dailyhunt.in/search?q={query}',
    'newsdog': 'https://newsdog.app/search?q={query}',
    'ucnews': 'https://ucnews.in/search?q={query}',
    'operanews': 'https://www.operanewsapp.com/search?q={query}',
    'sharechat': 'https://sharechat.com/search?q={query}',
    'mogo': 'https://mogo.tv/search?q={query}',
    'josh': 'https://josh.in/search?q={query}',
    'public': 'https://public.app/search?q={query}',
    'kooapp': 'https://www.kooapp.com/search?q={query}',
    'mailyolo': 'https://mailyolo.com/search?q={query}',
    'toffee': 'https://toffee.com/search?q={query}',
    'voot': 'https://www.voot.com/search?q={query}',
    'sonyliv': 'https://www.sonyliv.com/search?q={query}',
    'zee5': 'https://www.zee5.com/search?q={query}',
    'mxplayer': 'https://www.mxplayer.in/search?q={query}',
    'erosnow': 'https://www.erosnow.com/search?q={query}',
    'altbalaji': 'https://www.altbalaji.com/search?q={query}',
    'hoichoi': 'https://www.hoichoi.tv/search?q={query}',
    'aha': 'https://www.aha.video/search?q={query}',
    'bank': 'https://www.bankbazaar.com/search.html?q={query}',
    'perplexity': 'https://www.perplexity.ai/search?q={query}',
'gemini': 'https://gemini.google.com/app?query={query}',
'grok': 'https://grok.x.ai/?q={query}',
'claude': 'https://claude.ai/search?q={query}'

        }

    def _find_edge_path(self):
        """
        Auto-detect Microsoft Edge installation path
        Supports Windows, macOS, and Linux
        """
        system = platform.system()
        
        if system == 'Windows':
            # Windows paths to check
            windows_paths = [
                os.path.expandvars(r'%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe'),
                os.path.expandvars(r'%ProgramFiles%\Microsoft\Edge\Application\msedge.exe'),
            ]
            for path in windows_paths:
                if os.path.exists(path):
                    logger.info(f"Found Edge at: {path}")
                    return path
        
        elif system == 'Darwin':
            # macOS path
            mac_path = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
            if os.path.exists(mac_path):
                logger.info(f"Found Edge at: {mac_path}")
                return mac_path
        
        elif system == 'Linux':
            # Linux path
            linux_path = '/usr/bin/microsoft-edge'
            if os.path.exists(linux_path):
                logger.info(f"Found Edge at: {linux_path}")
                return linux_path
        
        logger.warning("Edge browser not found, will attempt 'msedge' command")
        return 'msedge'

    def _build_search_url(self, website, query):
        """
        Build search URL for given website and query
        
        Args:
            website (str): Website name (e.g., 'github', 'google')
            query (str): Search query
        
        Returns:
            str: Complete search URL with properly encoded query
        """
        # Normalize website name (lowercase, remove spaces)
        website_normalized = website.lower().strip()
        
        # Check if website has pre-configured pattern
        if website_normalized in self.website_patterns:
            pattern = self.website_patterns[website_normalized]
            # Use quote_plus to properly encode query (spaces as +, special chars as %xx)
            encoded_query = urllib.parse.quote_plus(query)
            return pattern.format(query=encoded_query)
        
        # Fallback for unknown websites
        # Try to build dynamic URL
        logger.info(f"Unknown website: {website}, building dynamic URL")
        
        # If it looks like a domain, use it directly
        if '.' in website_normalized:
            encoded_query = urllib.parse.quote_plus(query)
            return f"https://{website_normalized}/search?q={encoded_query}"
        
        # Default to Google if can't determine website
        logger.warning(f"Could not determine {website}, defaulting to Google")
        encoded_query = urllib.parse.quote_plus(query)
        return self.website_patterns['google'].format(query=encoded_query)

    def search(self, website, query):
        """
        Perform web search on specified website
        Opens Microsoft Edge with search results
        
        Args:
            website (str): Website to search on (e.g., 'github', 'google')
            query (str): Search query
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Build the search URL
            search_url = self._build_search_url(website, query)
            logger.info(f"Opening search: {website} - {query}")
            logger.info(f"URL: {search_url}")
            
            # Open Edge browser with search URL
            if platform.system() == 'Windows':
                subprocess.Popen([self.edge_path, search_url])
            else:
                subprocess.Popen([self.edge_path, search_url])
            
            return True
        
        except Exception as e:
            logger.error(f"Error opening search: {str(e)}")
            return False

    def open_website(self, website):
        """
        Open a website without search
        
        Args:
            website (str): Website URL or name
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If it's a known website key, use the base URL
            if website.lower() in self.website_patterns:
                base_urls = {
                    'youtube': 'https://www.youtube.com',
    'google': 'https://www.google.com',
    'gmail': 'https://mail.google.com',
    'facebook': 'https://www.facebook.com',
    'twitter': 'https://twitter.com',
    'instagram': 'https://www.instagram.com',
    'linkedin': 'https://www.linkedin.com',
    'github': 'https://github.com',
    'reddit': 'https://www.reddit.com',
    'amazon': 'https://www.amazon.in',
    'netflix': 'https://www.netflix.com',
    'spotify': 'https://open.spotify.com',
    'stackoverflow': 'https://stackoverflow.com',
    'wikipedia': 'https://www.wikipedia.org',
    'bing': 'https://www.bing.com',
    'x': 'https://x.com',
    'pinterest': 'https://www.pinterest.com',
    'medium': 'https://medium.com',
    'quora': 'https://www.quora.com',
    'ebay': 'https://www.ebay.com',
    'imdb': 'https://www.imdb.com',
    'duckduckgo': 'https://duckduckgo.com',
    'chatgpt': 'https://chatgpt.com',
    'whatsapp': 'https://web.whatsapp.com',
    'bank': 'https://www.bankbazaar.com',
    'hotstar': 'https://www.hotstar.com',
    'flipkart': 'https://www.flipkart.com',
    'cricbuzz': 'https://www.cricbuzz.com',
    'paytm': 'https://paytm.com',
    'timesofindia': 'https://timesofindia.indiatimes.com',
    'justdial': 'https://www.justdial.com',
    'jiosaavn': 'https://www.jiosaavn.com',
    'hindustantimes': 'https://www.hindustantimes.com',
    'indianexpress': 'https://indianexpress.com',
    'espncricinfo': 'https://www.espncricinfo.com',
    'bookmyshow': 'https://in.bookmyshow.com',
    'zeenews': 'https://zeenews.india.com',
    'snapchat': 'https://www.snapchat.com',
    'telegram': 'https://web.telegram.org',
    'magicbricks': 'https://www.magicbricks.com',
    'naukri': 'https://www.naukri.com',
    'shaadi': 'https://www.shaadi.com',
    'policybazaar': 'https://www.policybazaar.com',
    'myntra': 'https://www.myntra.com',
    'zomato': 'https://www.zomato.com',
    'swiggy': 'https://www.swiggy.com',
    'aajtak': 'https://www.aajtak.in',
    'ndtv': 'https://www.ndtv.com',
    'moneycontrol': 'https://www.moneycontrol.com',
    'gaana': 'https://gaana.com',
    'yatra': 'https://www.yatra.com',
    '99acres': 'https://www.99acres.com',
    'cleartrip': 'https://www.cleartrip.com',
    'makemytrip': 'https://www.makemytrip.com',
    'olx': 'https://www.olx.in',
    'carwale': 'https://www.carwale.com',
    'bikedekho': 'https://www.bikedekho.com',
    'housing': 'https://housing.com',
    'bigbasket': 'https://www.bigbasket.com',
    'grofers': 'https://www.blinkit.com',
    'reliancedigital': 'https://www.reliancedigital.in',
    'vijaysales': 'https://www.vijaysales.com',
    'croma': 'https://www.croma.com',
    'indiamart': 'https://www.indiamart.com',
    'tradeindia': 'https://www.tradeindia.com',
    'sulekha': 'https://www.sulekha.com',
    'mouthshut': 'https://www.mouthshut.com',
    'whatmobile': 'https://whatmobile.net.in',
    'hindisahityadarpan': 'https://hindisahityadarpan.in',
    'abhiyojana': 'https://abhiyojana.co.in',
    'pib': 'https://pib.gov.in',
    'incometaxindia': 'https://incometaxindia.gov.in',
    'upi': 'https://www.npci.org.in',
    'bhimgov': 'https://bhimgov.in',
    'umang': 'https://web.umang.gov.in',
    'digilocker': 'https://digilocker.gov.in',
    'mygov': 'https://www.mygov.in',
    'india': 'https://www.india.gov.in',
    'epfindia': 'https://www.epfindia.gov.in',
    'esi': 'https://www.esic.gov.in',
    'passportindia': 'https://www.passportindia.gov.in',
    'irctc': 'https://www.irctc.co.in',
    'indianrail': 'https://indianrail.gov.in',
    'airindia': 'https://www.airindia.in',
    'raaga': 'https://www.raaga.com',
    'hungama': 'https://www.hungama.com',
    'wynk': 'https://wynk.in',
    'jio': 'https://www.jio.com',
    'airtel': 'https://www.airtel.in',
    'vi': 'https://www.myvi.in',
    'bsnl': 'https://www.bsnl.co.in',
    'icicibank': 'https://www.icicibank.com',
    'hdfcbank': 'https://www.hdfcbank.com',
    'sbibank': 'https://sbi.co.in',
    'axisbank': 'https://www.axisbank.com',
    'kotak': 'https://www.kotak.com',
    'yesbank': 'https://www.yesbank.in',
    'federalbank': 'https://www.federalbank.co.in',
    'ucobank': 'https://www.ucobank.com',
    'canarabank': 'https://canarabank.com',
    'iifl': 'https://www.iifl.com',
    'bajajfinserv': 'https://www.bajajfinserv.in',
    'tatacapital': 'https://www.tatacapital.com',
    'lendenclub': 'https://www.lendenclub.com',
    'groww': 'https://groww.in',
    'zerodha': 'https://zerodha.com',
    'upstox': 'https://upstox.com',
    'angelone': 'https://www.angelone.in',
    '5paisa': 'https://www.5paisa.com',
    'dhan': 'https://dhan.co',
    'icicidirect': 'https://www.icicidirect.com',
    'sharekhan': 'https://www.sharekhan.com',
    'motilaloswal': 'https://www.motilaloswal.com',
    'choiceindia': 'https://www.choiceindia.com',
    'rkglobal': 'https://www.rkglobal.net',
    'indmoney': 'https://www.indmoney.com',
    'smallcase': 'https://www.smallcase.com',
    'fisdom': 'https://www.fisdom.com',
    'bajajbroking': 'https://www.bajajbroking.in',
    'hdfcsec': 'https://www.hdfcsec.com',
    'kfintech': 'https://www.kfintech.com',
    'nseindia': 'https://www.nseindia.com',
    'bseindia': 'https://www.bseindia.com',
    'mcxindia': 'https://www.mcxindia.com',
    'moneybhai': 'https://moneybhai.moneycontrol.com',
    'tickertape': 'https://www.tickertape.in',
    'stockedge': 'https://stockedge.com',
    'screener': 'https://www.screener.in',
    'trendlyne': 'https://trendlyne.com',
    'tradingview': 'https://www.tradingview.com',
    'investing': 'https://www.investing.com',
    'yahoofinance': 'https://in.finance.yahoo.com',
    'economictimes': 'https://economictimes.indiatimes.com',
    'livemint': 'https://www.livemint.com',
    'financialexpress': 'https://www.financialexpress.com',
    'businesstoday': 'https://www.businesstoday.in',
    'business-standard': 'https://www.business-standard.com',
    'firstpost': 'https://www.firstpost.com',
    'scroll': 'https://scroll.in',
    'theprint': 'https://theprint.in',
    'opindia': 'https://www.opindia.com',
    'republicworld': 'https://www.republicworld.com',
    'news18': 'https://www.news18.com',
    'tv9telugu': 'https://tv9telugu.com',
    'sakshi': 'https://www.sakshi.com',
    'andhrajyothy': 'https://www.andhrajyothy.com',
    'manatelangana': 'https://www.manatelangana.com',
    'greatandhra': 'https://www.greatandhra.com',
    'idlebrain': 'https://www.idlebrain.com',
    '123telugu': 'https://www.123telugu.com',
    'filmibeat': 'https://www.filmibeat.com',
    'bollywoodhungama': 'https://www.bollywoodhungama.com',
    'pinkvilla': 'https://www.pinkvilla.com',
    'koimoi': 'https://www.koimoi.com',
    'indiaforums': 'https://www.indiaforums.com',
    'tellychakkar': 'https://www.tellychakkar.com',
    'iwmbuzz': 'https://www.iwmbuzz.com',
    'gossipaddict': 'https://gossipaddict.com',
    'missmalini': 'https://www.missmalini.com',
    'jagran': 'https://www.jagran.com',
    'dainikbhaskar': 'https://www.bhaskar.com',
    'amarujala': 'https://www.amarujala.com',
    'navbharattimes': 'https://navbharattimes.indiatimes.com',
    'abhiyaan': 'https://abhiyaan.com',
    'livehindustan': 'https://www.livehindustan.com',
    'jansatta': 'https://www.jansatta.com',
    'rajasthanpatrika': 'https://www.patrika.com',
    'bhaskar': 'https://www.bhaskar.com',
    'divyabhaskar': 'https://www.divyabhaskar.co.in',
    'inshorts': 'https://www.inshorts.com',
    'dailyhunt': 'https://www.dailyhunt.in',
    'newsdog': 'https://newsdog.app',
    'ucnews': 'https://ucnews.in',
    'operanews': 'https://www.operanewsapp.com',
    'sharechat': 'https://sharechat.com',
    'mogo': 'https://mogo.tv',
    'josh': 'https://josh.in',
    'public': 'https://public.app',
    'kooapp': 'https://www.kooapp.com',
    'mailyolo': 'https://mailyolo.com',
    'toffee': 'https://toffee.com',
    'voot': 'https://www.voot.com',
    'sonyliv': 'https://www.sonyliv.com',
    'zee5': 'https://www.zee5.com',
    'mxplayer': 'https://www.mxplayer.in',
    'erosnow': 'https://www.erosnow.com',
    'altbalaji': 'https://www.altbalaji.com',
    'hoichoi': 'https://www.hoichoi.tv',
    'aha': 'https://www.aha.video',
                    'perplexity': 'https://www.perplexity.ai',
                    'gemini': 'https://gemini.google.com',
                    'grok': 'https://grok.x.ai',
                    'claude': 'https://claude.ai',
                    'chatgpt': 'https://chat.openai.com',
                }
                url = base_urls.get(website.lower(), website)
            else:
                # Assume it's a URL
                if not website.startswith('http'):
                    url = 'https://' + website
                else:
                    url = website
            
            logger.info(f"Opening website: {url}")
            subprocess.Popen([self.edge_path, url])
            return True
        
        except Exception as e:
            logger.error(f"Error opening website: {str(e)}")
            return False


# Test the handler if run directly
if __name__ == "__main__":
    handler = EdgeSearchHandler()
    
    # Test search on different websites
    print("Testing EdgeSearchHandler...")
    print(f"Edge path: {handler.edge_path}")
    
    # Test URL building
    test_cases = [
        ('google', 'python programming'),
        ('github', 'react components'),
        ('stackoverflow', 'javascript error'),
        ('wikipedia', 'artificial intelligence'),
        ('youtube', 'machine learning tutorial'),
    ]
    
    print("\nTest URLs:")
    for website, query in test_cases:
        url = handler._build_search_url(website, query)
        print(f"  {website} + '{query}': {url}")
