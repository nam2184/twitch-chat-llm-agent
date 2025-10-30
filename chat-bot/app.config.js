import 'dotenv/config';

export default ({ config }) => ({
  ...config,
  extra: {
    apiUrl: process.env.API_BASE_URL,
    heUrl: process.env.HE_BASE_URL,
  },
});

