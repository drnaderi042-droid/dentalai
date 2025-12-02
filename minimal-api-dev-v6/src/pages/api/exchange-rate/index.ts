import type { NextApiRequest, NextApiResponse } from 'next';

import axios from 'axios';
import * as cheerio from 'cheerio';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Fetch exchange rate from bonbast.com
async function fetchBonbastRate(): Promise<{ usdToIrr: number; eurToIrr: number } | null> {
  try {
    console.log('[Exchange Rate] Fetching from bonbast.com...');
    
    const response = await axios.get('https://bonbast.com/', {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      },
      timeout: 10000,
    });

    const $ = cheerio.load(response.data);
    
    // Extract USD to IRR rate (buy price)
    // Bonbast uses specific classes for prices
    const usdSellText = $('td[data-market-row="usd"] td[data-sell]').text().trim();
    const eurSellText = $('td[data-market-row="eur"] td[data-sell]').text().trim();
    
    console.log('[Exchange Rate] Raw USD:', usdSellText);
    console.log('[Exchange Rate] Raw EUR:', eurSellText);

    // Parse the price (remove commas and convert to number)
    const usdToIrr = parseFloat(usdSellText.replace(/,/g, ''));
    const eurToIrr = parseFloat(eurSellText.replace(/,/g, ''));

    if (!usdToIrr || Number.isNaN(usdToIrr)) {
      throw new Error('Failed to parse USD rate from bonbast.com');
    }

    console.log('[Exchange Rate] Parsed USD:', usdToIrr);
    console.log('[Exchange Rate] Parsed EUR:', eurToIrr || 'N/A');

    return {
      usdToIrr: usdToIrr / 10, // Convert Rials to Tomans
      eurToIrr: eurToIrr ? eurToIrr / 10 : 0,
    };
  } catch (error) {
    console.error('[Exchange Rate] Error fetching from bonbast:', error);
    return null;
  }
}

// Get or update exchange rate
async function getExchangeRate() {
  try {
    // Check if we have a recent rate (less than 6 hours old)
    const existingRate = await prisma.exchangeRate.findFirst({
      where: {
        expiresAt: {
          gt: new Date(),
        },
      },
      orderBy: {
        fetchedAt: 'desc',
      },
    });

    if (existingRate) {
      console.log('[Exchange Rate] Using cached rate');
      return existingRate;
    }

    // Fetch new rate from bonbast
    console.log('[Exchange Rate] Fetching new rate...');
    const bonbastRate = await fetchBonbastRate();

    if (!bonbastRate) {
      // If bonbast fails, use fallback or last known rate
      const lastRate = await prisma.exchangeRate.findFirst({
        orderBy: {
          fetchedAt: 'desc',
        },
      });

      if (lastRate) {
        console.log('[Exchange Rate] Using last known rate as fallback');
        return lastRate;
      }

      // Ultimate fallback - use a default rate
      console.log('[Exchange Rate] Using default fallback rate');
      return {
        id: 'fallback',
        usdToIrr: 70000, // 70,000 Toman (current approximate rate)
        eurToIrr: 75000, // 75,000 Toman (current approximate rate)
        source: 'fallback',
        fetchedAt: new Date(),
        expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000), // 6 hours
        createdAt: new Date(),
      };
    }

    // Save new rate to database
    const newRate = await prisma.exchangeRate.create({
      data: {
        usdToIrr: bonbastRate.usdToIrr,
        eurToIrr: bonbastRate.eurToIrr,
        source: 'bonbast',
        fetchedAt: new Date(),
        expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000), // Expires in 6 hours
      },
    });

    console.log('[Exchange Rate] New rate saved:', newRate);
    return newRate;
  } catch (error) {
    console.error('[Exchange Rate] Error:', error);
    throw error;
  }
}

// API Handler
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const rate = await getExchangeRate();

    return res.status(200).json({
      success: true,
      data: {
        usd_to_irr: rate.usdToIrr,
        eur_to_irr: rate.eurToIrr,
        source: rate.source,
        fetched_at: rate.fetchedAt,
        expires_at: rate.expiresAt,
      },
    });
  } catch (error) {
    console.error('[Exchange Rate API] Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to fetch exchange rate',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

