import { parsePhoneNumber } from 'react-phone-number-input';

import { countries } from 'src/assets/data/countries';

import { countryNamesFa } from './country-names-fa';

// ----------------------------------------------------------------------

export function getCountryCode(inputValue, countryCode) {
  if (inputValue) {
    const phoneNumber = parsePhoneNumber(inputValue);

    if (phoneNumber) {
      return phoneNumber?.country;
    }
  }

  return countryCode ?? 'IR'; // Default to Iran
}

// ----------------------------------------------------------------------

export function getCountry(countryCode) {
  const option = countries.filter((country) => country.code === countryCode)[0];
  return option;
}

export function applyFilter({ inputData, query }) {
  if (query) {
    return inputData.filter(
      (country) => {
        const countryNameFa = countryNamesFa[country.code] || '';
        return (
          country.label.toLowerCase().indexOf(query.toLowerCase()) !== -1 ||
          country.code.toLowerCase().indexOf(query.toLowerCase()) !== -1 ||
          country.phone.toLowerCase().indexOf(query.toLowerCase()) !== -1 ||
          countryNameFa.indexOf(query) !== -1
        );
      }
    );
  }

  return inputData;
}
