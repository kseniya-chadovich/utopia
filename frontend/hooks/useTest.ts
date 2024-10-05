"use client";

import { useEffect, useState } from 'react';
import axios from 'axios';

const useTest = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get("http://localhost:8080/");
        setData(response.data);
      } catch (error) {
        setError(error as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { data, error, loading };
};

export default useTest;